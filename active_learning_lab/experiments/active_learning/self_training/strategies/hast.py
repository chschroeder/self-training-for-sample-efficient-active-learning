import logging
import faiss

import numpy as np

from small_text.data.sampling import _get_class_histogram

from active_learning_lab.experiments.active_learning.self_training.strategies.base import (
    SelfTrainingStrategy,
    SelfTrainingStatus,
    SelfTrainingOverallResults
)

from active_learning_lab.experiments.active_learning.self_training.utils.index import _initialize_index, _get_hard_label_agreement_scores
from active_learning_lab.utils.time import measure_time_context


logger = logging.getLogger(__name__)


def _get_class_weights(y_pred_new, num_classes):
    class_hist = _get_class_histogram(y_pred_new, num_classes)
    class_hist_mean = int(class_hist.sum() / num_classes)

    observed_number_class_instances = class_hist[y_pred_new]
    distance_from_expected_number_of_class_instances = class_hist_mean - class_hist[y_pred_new]

    class_weight = np.divide(distance_from_expected_number_of_class_instances,
                             observed_number_class_instances,
                             out=np.ones_like(observed_number_class_instances, dtype=np.float64),
                             where=observed_number_class_instances != 0)

    class_weight = 10 / (1 + np.exp(-1 * class_weight))

    return class_weight


class HAST(SelfTrainingStrategy):

    def __init__(self,
                 self_training_iterations=1,
                 agreement_threshold=0.5,
                 filter_threshold=0.5,
                 knn_k=15,
                 subsample_size=16384,
                 use_class_weights=False,  # enables alpha if True
                 labeled_to_unlabeled_factor=1.0,  # beta
                 mini_batch_size=32):
        """
        Parameters
        ----------

        """
        self.self_training_iterations = self_training_iterations
        self.agreement_threshold = agreement_threshold
        self.filter_threshold = filter_threshold
        self.knn_k = knn_k

        self.subsample_size = subsample_size

        self.use_class_weights = use_class_weights
        self.labeled_to_unlabeled_factor = labeled_to_unlabeled_factor

        self.mini_batch_size = mini_batch_size

    def train(self, clf, dataset, y, indices_unlabeled, indices_labeled, indices_valid, test_set=None):

        embeddings, y_train_pred_proba_before = clf.embed(dataset, return_proba=True)
        y_test_pred_proba_before = clf.predict_proba(test_set)

        indices_pseudo_labeled = np.copy(indices_labeled)
        y_pseudo_labeled = np.copy(y)

        results = []

        weights = np.array([1.0] * indices_labeled.shape[0])

        for t in range(self.self_training_iterations):
            result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, \
            y_train_pred_proba, y_test_pred_proba, weights = \
                self._self_training_iteration(
                    clf,
                    dataset,
                    indices_labeled,
                    indices_pseudo_labeled,
                    y_pseudo_labeled,
                    indices_unlabeled,
                    indices_valid,
                    test_set,
                    weights
                )
            results.append(result)

        y_train_pred_proba_after = y_train_pred_proba
        y_test_pred_proba_after = y_test_pred_proba

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
                                          y_test_pred_proba_after)

    def _self_training_iteration(self,
                                 clf,
                                 dataset,
                                 indices_labeled,
                                 indices_pseudo_labeled,
                                 y_pseudo_labeled,
                                 indices_unlabeled,
                                 indices_valid,
                                 test_set,
                                 weights):

        with measure_time_context() as pseudo_label_query_time:
            indices_new, y_pred_proba_new, previous_scores = self.get_pseudo_labeled_dataset(clf,
                                                                                             dataset,
                                                                                             indices_unlabeled)


            indices_thresholded = previous_scores > self.agreement_threshold

            indices_new = indices_new[indices_thresholded]
            y_pred_proba_new = y_pred_proba_new[indices_thresholded]

        if indices_new.shape[0] == 0:
            logger.info(f'[self-training-strategy] No pseudo-labels exceeded threshold. '
                        f'No self-training after thresholding.')
            validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]

            new_weights = np.empty((0,))
            return self._abort(clf,
                               dataset,
                               test_set,
                               validation_set,
                               indices_labeled,
                               indices_unlabeled,
                               indices_new,
                               indices_pseudo_labeled,
                               SelfTrainingStatus.FAILURE_NO_PSEUDO_LABELS_AFTER_THRESHOLDING,
                               pseudo_label_query_time(),
                               train_weights=weights / weights.sum()) + (new_weights, )
        else:
            indices_pseudo_labeled = np.append(indices_pseudo_labeled, indices_new)
            y_pred_new = np.argmax(y_pred_proba_new, axis=1)
            y_pseudo_labeled = np.append(y_pseudo_labeled, y_pred_new)

            indices_unlabeled = set(indices_unlabeled.tolist())
            indices_unlabeled = indices_unlabeled - set(indices_new.tolist())
            indices_unlabeled = np.array(list(indices_unlabeled))

            validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]

            if self.use_class_weights:
                class_weight = _get_class_weights(y_pred_new, clf.num_classes)
            else:
                class_weight = np.ones(y_pred_new.shape)

            new_weights = class_weight * self.labeled_to_unlabeled_factor
            weights = np.append(weights, new_weights)

            dataset_copy = dataset[indices_pseudo_labeled].clone()
            dataset_copy.y = y_pseudo_labeled

            with measure_time_context() as update_time:
                self._train(clf,
                            dataset_copy,
                            validation_set,
                            indices_labeled.shape[0],
                            indices_pseudo_labeled.shape[0] - indices_labeled.shape[0],
                            train_weights=weights / weights.sum())

            result, y_train_pred_proba, y_test_pred_proba = self._evaluate(
                clf,
                dataset,
                test_set,
                indices_labeled,
                indices_pseudo_labeled,
                indices_new,
                SelfTrainingStatus.SUCCESS,
                pseudo_label_query_time(),
                update_time()
            )

            return result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, \
                   y_train_pred_proba, y_test_pred_proba, new_weights

    def get_pseudo_labeled_dataset(self, clf, dataset, indices_unlabeled):

        if self.subsample_size is None:
            indices_subsampled = indices_unlabeled
        else:
            indices_subsampled = np.random.choice(indices_unlabeled,
                                                  min(self.subsample_size, indices_unlabeled.shape[0]),
                                                  replace=False)

        embeddings, y_pred_proba = clf.embed(dataset[indices_subsampled], return_proba=True)
        y_pred = np.argmax(y_pred_proba, axis=1)

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        index = _initialize_index(embeddings, embeddings)

        scores, knn_pred = _get_hard_label_agreement_scores(index,
                                                            indices_subsampled,
                                                            embeddings,
                                                            y_pred,
                                                            clf.num_classes,
                                                            mini_batch_size=self.mini_batch_size,
                                                            knn_k=self.knn_k)
        del index

        knn_agreement = (knn_pred == y_pred)
        sample_certainty = (y_pred_proba.max(axis=1) > self.filter_threshold)
        logging.info(f'knn_agreement: {knn_agreement.sum()} / {indices_subsampled.shape[0]}, '
                     f'sample_certainty: {sample_certainty.sum()} / {indices_subsampled.shape[0]}'
        )

        indices_verified = knn_agreement & sample_certainty

        return indices_subsampled[indices_verified], y_pred_proba[indices_verified], scores[indices_verified]
