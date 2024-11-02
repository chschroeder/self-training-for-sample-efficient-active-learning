import logging
import numpy as np

from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from small_text.data.sampling import _get_class_histogram

from active_learning_lab.utils.pytorch import free_resources_fix
from active_learning_lab.experiments.active_learning.self_training.strategies.base import (
get_empty_iteration_result,
    get_iteration_results,
    SelfTrainingOverallResults,
    SelfTrainingStatus,
    SelfTrainingStrategy
)
from active_learning_lab.utils.time import measure_time_context


logger = logging.getLogger(__name__)


def get_self_training_samples(embeddings, y, uncertainty, num_classes, num_clusters, num_sampled_clusters,
                              num_samples_per_cluster, beta):

    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    cluster_indices = kmeans.fit_predict(embeddings, sample_weight=uncertainty)

    region_uncertainties = np.empty((0,), dtype=float)
    for i in range(kmeans.cluster_centers_.shape[0]):
        indices = np.argwhere(cluster_indices == i)[:, 0]

        average_cluster_uncertainty = np.mean(uncertainty[indices])
        inter_class_diversity = entropy(_get_class_histogram(y[indices], num_classes, normalize=True))

        region_uncertainties = np.append(region_uncertainties, average_cluster_uncertainty + beta*inter_class_diversity)

    argwhere_per_class = [np.argwhere(cluster_indices == i)[:, 0] for i in range(kmeans.cluster_centers_.shape[0])]
    num_samples_per_cluster_adjusted = get_num_samples_per_cluster(region_uncertainties,
                                                                   argwhere_per_class,
                                                                   num_sampled_clusters,
                                                                   num_samples_per_cluster)

    resulting_indices = np.empty((0,), dtype=int)
    for i in np.argpartition(region_uncertainties, num_sampled_clusters)[:num_sampled_clusters]:
        indices = argwhere_per_class[i]

        # sample by minimum uncertainty (easy samples)
        if indices.shape[0] > num_samples_per_cluster_adjusted[i]:
            sampled_indices = np.argpartition(uncertainty[indices],
                                              num_samples_per_cluster_adjusted[i])[:num_samples_per_cluster_adjusted[i]]
        else:
            sampled_indices = np.s_[:]
        resulting_indices = np.append(resulting_indices, indices[sampled_indices])

    logger.info(f'[self-training-strategy] num_self_training_examples={resulting_indices.shape[0]}, '
                f'avg_confidence={1-np.mean(uncertainty[sampled_indices])}')
    return resulting_indices


def get_num_samples_per_cluster(region_uncertainties, argwhere_per_class, num_sampled_clusters, num_samples_per_cluster):
    num_clusters = len(region_uncertainties)

    result = np.zeros((num_clusters,), dtype=int)

    carryover = 0
    region_uncertainties_argpartitioned = np.argpartition(region_uncertainties, num_sampled_clusters)
    for i in region_uncertainties_argpartitioned[:num_sampled_clusters]:
        desired_number_samples = num_samples_per_cluster + carryover
        num_to_samples = min(desired_number_samples, argwhere_per_class[i].shape[0])
        result[i] = num_to_samples
        carryover = desired_number_samples - num_to_samples

    if carryover > 0:
        remaining_samples = np.array([ind.shape[0] for ind in argwhere_per_class]) - result
        if remaining_samples.sum() <= 0:
            raise ValueError('Could not sample enough instances per cluster: carryover > 0')
        else:
            classes = []
            for i, n in enumerate(remaining_samples):
                classes += [i] * n
            for i in np.random.choice(classes, carryover):
                result[i] += 1

    return result


def get_momentum_coefficient(iteration, total_iterations, momentum_l, momentum_h):
    iteration_rel = iteration / total_iterations
    return (1 - iteration_rel) * momentum_l + iteration_rel * momentum_h


class AcTune(SelfTrainingStrategy):

    def __init__(self,
                 num_clusters=10, # K
                 num_sampled_clusters=5,  # M
                 num_samples_per_cluster=5,  # floor(B/M) where B=query_size
                 self_training_iterations=1,  # T
                 momentum_l=0.8,
                 momentum_h=0.9,
                 beta=0.5,
                 gamma=0.6,
                 lmbda=1,
                 mini_batch_size=32):
        """
        Parameters
        ----------

        """
        self.num_clusters = num_clusters
        self.num_sampled_clusters = num_sampled_clusters
        # TODO: if query_size is not divisible without remainder by num_samples_per_cluster,
        #       we query less than query_size in total
        self.num_samples_per_cluster = num_samples_per_cluster  # floor(B/M)

        self.self_training_iterations = self_training_iterations
        self.momentum_l = momentum_l
        self.momentum_h = momentum_h

        self.beta = beta
        self.gamma = gamma
        self.lmbda = lmbda

        self.mini_batch_size = mini_batch_size

    def train(self, clf, dataset, y, indices_unlabeled, indices_labeled, indices_valid, test_set=None):

        y_train_pred_proba_before = clf.predict_proba(dataset)
        y_test_pred_proba_before = clf.predict_proba(test_set)

        y_pred_proba_prev = None

        indices_pseudo_labeled = np.copy(indices_labeled)
        y_pseudo_labeled = np.copy(dataset.y[indices_labeled])

        results = []

        for t in range(self.self_training_iterations):
            result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, y_pred_proba, y_pred_proba_prev = self._self_training_iteration(
                clf,
                dataset,
                indices_labeled,
                indices_pseudo_labeled,
                indices_unlabeled,
                indices_valid, t, test_set,
                y_pred_proba_prev,
                y_pseudo_labeled)

            results.append(result)
            free_resources_fix()

        y_train_pred_proba_after = y_pred_proba
        y_test_pred_proba_after = clf.predict_proba(test_set)

        return SelfTrainingOverallResults(results,
                                          indices_labeled.shape[0],
                                          indices_pseudo_labeled.shape[0],
                                          dataset.y,
                                          np.argmax(y_train_pred_proba_before, axis=1),
                                          y_train_pred_proba_before,
                                          np.argmax(y_train_pred_proba_after, axis=1),
                                          y_train_pred_proba_after,
                                          test_set.y,
                                          np.argmax(y_test_pred_proba_before, axis=1),
                                          y_test_pred_proba_before,
                                          np.argmax(y_test_pred_proba_after, axis=1),
                                          y_test_pred_proba_after
                                          )

    def _self_training_iteration(self, clf, dataset, indices_labeled, indices_pseudo_labeled, indices_unlabeled,
                                 indices_valid, t, test_set, y_pred_proba_prev, y_pseudo_labeled):

        with measure_time_context() as pseudo_label_query_time:
            embeddings, y_pred_proba = clf.embed(dataset, return_proba=True)
            embeddings = normalize(embeddings, axis=1)
            if y_pred_proba_prev is None:
                y_pred_proba_prev = y_pred_proba

            uncertainty = np.apply_along_axis(entropy, 1, y_pred_proba) / np.log(clf.num_classes)

            mt = get_momentum_coefficient(t, self.self_training_iterations, self.momentum_l, self.momentum_h)
            logger.info(f'[self-training-strategy] Momentum coefficient={mt}')

            y_pred_proba = mt * y_pred_proba + (1 - mt) * y_pred_proba_prev
            y_pred = np.argmax(y_pred_proba, axis=1)

            indices_thresholded = indices_unlabeled[np.amax(y_pred_proba[indices_unlabeled], axis=1) > self.gamma]
            if indices_thresholded.shape[0] == 0:
                logger.info(f'[self-training-strategy] No pseudo-labels exceeded threshold. '
                            f'No self-training after verification step.')
                return get_empty_iteration_result(
                    clf.num_classes,
                    indices_labeled.shape[0],
                    SelfTrainingStatus.FAILURE_NO_PSEUDO_LABELS_AFTER_THRESHOLDING,
                    pseudo_label_query_time()
                ), indices_unlabeled, indices_pseudo_labeled, np.zeros((indices_pseudo_labeled.shape[0],)), \
                       np.zeros((len(dataset), clf.num_classes)), np.zeros((len(test_set), clf.num_classes))
            elif indices_thresholded.shape[0] < self.num_clusters:
                indices_new = indices_thresholded
            else:
                indices_new = get_self_training_samples(embeddings[indices_thresholded],
                                                        y_pred[indices_thresholded],
                                                        uncertainty[indices_thresholded],
                                                        clf.num_classes,
                                                        self.num_clusters,
                                                        self.num_sampled_clusters,
                                                        self.num_samples_per_cluster,
                                                        self.beta)

            if np.unique(dataset.y[indices_new]).shape[0] == 1:
                logger.info(f'[self-training-strategy]  Encountered only a single class. '
                            f'No self-training before verification step.')
                return get_empty_iteration_result(
                    clf.num_classes,
                    indices_labeled.shape[0],
                    SelfTrainingStatus.FAILURE_ONE_CLASS,
                    pseudo_label_query_time()
                ), indices_unlabeled, indices_pseudo_labeled, np.zeros((indices_pseudo_labeled.shape[0],)), \
                       np.zeros((len(dataset), clf.num_classes)), np.zeros((len(test_set), clf.num_classes))

        indices_pseudo_labeled = np.append(indices_pseudo_labeled, indices_new)
        y_pseudo_labeled = np.append(y_pseudo_labeled, y_pred[indices_new])

        indices_unlabeled = set(indices_unlabeled.tolist())
        indices_unlabeled = indices_unlabeled - set(indices_new.tolist())
        indices_unlabeled = np.array(list(indices_unlabeled))

        dataset_new = dataset[indices_pseudo_labeled].clone()
        dataset_new.y = y_pseudo_labeled

        logger.info(f'[self-training-strategy] Self-Training label distribution '
                    f'{_get_class_histogram(dataset.y[indices_pseudo_labeled], clf.num_classes)} .')

        weights = np.array([1. / indices_labeled.shape[0]] * indices_labeled.shape[0])
        num_only_pseudo_labeled = indices_pseudo_labeled.shape[0] - indices_labeled.shape[0]
        weights = np.append(weights,
                            np.array([self.lmbda / num_only_pseudo_labeled] * num_only_pseudo_labeled))

        validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]
        with measure_time_context() as update_time:
            self._train(clf,
                        dataset_new,
                        validation_set,
                        indices_labeled.shape[0],
                        indices_pseudo_labeled.shape[0] - indices_labeled.shape[0],
                        train_weights=weights)

        y_pred_proba_prev = y_pred_proba
        with measure_time_context() as evaluation_time:
            y_train_pred_proba = clf.predict_proba(dataset)
            y_train_pred = np.argmax(y_train_pred_proba, 1)
            y_test_pred_proba = clf.predict_proba(test_set)

        result = get_iteration_results(clf.num_classes,
                                       indices_labeled.shape[0],
                                       indices_pseudo_labeled.shape[0],
                                       SelfTrainingStatus.SUCCESS,
                                       pseudo_label_query_time(),
                                       update_time(),
                                       evaluation_time(),
                                       dataset[indices_new].y,
                                       y_train_pred[indices_new],
                                       y_train_pred_proba[indices_new],
                                       dataset[indices_pseudo_labeled].y,
                                       y_train_pred[indices_pseudo_labeled],
                                       y_train_pred_proba[indices_pseudo_labeled],
                                       dataset.y,
                                       y_train_pred,
                                       y_train_pred_proba,
                                       test_set.y,
                                       np.argmax(y_test_pred_proba, 1),
                                       y_test_pred_proba)

        return result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, y_pred_proba, y_pred_proba_prev
