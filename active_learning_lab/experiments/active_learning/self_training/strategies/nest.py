import faiss
import logging

import numpy as np

from small_text.data.sampling import _get_class_histogram

from active_learning_lab.experiments.active_learning.self_training.strategies.base import (
    SelfTrainingStrategy,
    SelfTrainingStatus,
    SelfTrainingOverallResults
)
from active_learning_lab.experiments.active_learning.self_training.utils.index import _get_kl_scores, _initialize_index
from active_learning_lab.utils.time import measure_time_context


logger = logging.getLogger(__name__)


class NEST(SelfTrainingStrategy):

    def __init__(self,
                 self_training_iterations=3,   # T
                 self_training_threshold=0.9,  # gamma
                 previous_round_weight=0.5,    # m
                 num_pseudo_labels=None,       # b
                 labeled_loss_weight=0.5,      # lambda
                 divergence_beta=0.1,          # beta
                 ablation_unlabeled_divergence_weight=1.0,
                 subsample_size=16384,
                 knn_k=7,
                 c=10,
                 mini_batch_size=32,
                 reuse_model=True):

        self.self_training_iterations = self_training_iterations
        self.self_training_threshold = self_training_threshold
        self.previous_round_weight = previous_round_weight

        self.num_pseudo_labels = num_pseudo_labels
        self.labeled_loss_weight = labeled_loss_weight
        self.divergence_beta = divergence_beta
        self.mini_batch_size = mini_batch_size

        self.subsample_size = subsample_size
        self.knn_k = knn_k
        self.c = c

        self.reuse_model = reuse_model

        self.indices_labeled = set()
        self.index = None

    def train(self, clf, dataset, y, indices_unlabeled, indices_labeled, indices_valid, test_set=None):

        self.num_pseudo_labels = self.c * indices_labeled.shape[0]

        y_train_pred_proba_before = clf.predict_proba(dataset)
        y_test_pred_proba_before = clf.predict_proba(test_set)

        indices_pseudo_labeled = np.copy(indices_labeled)
        y_pseudo_labeled = np.copy(y)

        results = []

        shape = indices_unlabeled.shape if self.subsample_size is None else (self.subsample_size,)
        if indices_unlabeled.shape[0] < shape[0]:
            shape = (indices_unlabeled.shape[0],)

        previous_scores = np.zeros(shape)

        for t in range(self.self_training_iterations):
            result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, \
            y_train_pred_proba, y_test_pred_proba, previous_scores = \
                self._self_training_iteration(
                    clf,
                    dataset,
                    indices_labeled,
                    indices_pseudo_labeled,
                    y_pseudo_labeled,
                    indices_unlabeled,
                    indices_valid,
                    test_set,
                    previous_scores
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
                                 previous_scores):

        with measure_time_context() as pseudo_label_query_time:
            indices_new, y_pred_proba_new, previous_scores = self.get_pseudo_labeled_dataset(clf,
                                                                                             dataset,
                                                                                             indices_unlabeled,
                                                                                             indices_labeled,
                                                                                             previous_scores)

        is_above_threshold = np.amax(y_pred_proba_new, axis=1) > self.self_training_threshold
        indices_new = indices_new[is_above_threshold]
        y_pred_proba_new = y_pred_proba_new[is_above_threshold]
        ###

        if indices_new.shape[0] == 0:
            logger.info(f'[self-training-strategy] No pseudo-labels exceeded threshold. '
                        f'No self-training after thresholding.')
            validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]

            train_weights = np.array([self.labeled_loss_weight] * indices_labeled.shape[0] +
                                     [1-self.labeled_loss_weight] * (indices_pseudo_labeled.shape[0] - indices_labeled.shape[0]))
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
                               train_weights=train_weights) + (previous_scores, )
        else:
            indices_pseudo_labeled = np.append(indices_pseudo_labeled, indices_new)
            y_pseudo_labeled = np.append(y_pseudo_labeled, np.argmax(y_pred_proba_new, axis=1))

            indices_unlabeled = set(indices_unlabeled.tolist())
            indices_unlabeled = indices_unlabeled - set(indices_new.tolist())
            indices_unlabeled = np.array(list(indices_unlabeled))

            validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]
            train_weights = np.array([self.labeled_loss_weight] * indices_labeled.shape[0] +
                                     [1 - self.labeled_loss_weight] * (indices_pseudo_labeled.shape[0] - indices_labeled.shape[0]))

            dataset_copy = dataset[indices_pseudo_labeled].clone()
            dataset_copy.y = y_pseudo_labeled

            with measure_time_context() as update_time:
                self._train(clf,
                            dataset_copy,
                            validation_set,
                            indices_labeled.shape[0],
                            indices_pseudo_labeled.shape[0] - indices_labeled.shape[0],
                            train_weights=train_weights)

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

            print('## self-training labels ', _get_class_histogram(dataset.y[indices_pseudo_labeled], clf.num_classes))

            return result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, \
                   y_train_pred_proba, y_test_pred_proba, previous_scores

    def get_pseudo_labeled_dataset(self, clf, dataset, indices_unlabeled, indices_labeled, previous_scores):

        if self.subsample_size is None:
            indices_subsampled = indices_unlabeled
        else:
            indices_subsampled = np.random.choice(indices_unlabeled,
                                                  min(self.subsample_size, indices_unlabeled.shape[0]),
                                                  replace=False)

        embeddings, y_pred_proba = clf.embed(dataset[indices_subsampled], return_proba=True)

        embeddings_labeled, y_pred_proba_labeled = clf.embed(dataset[indices_labeled], return_proba=True)


        embeddings = embeddings.astype(np.float32)
        embeddings_labeled = embeddings_labeled.astype(np.float32)

        faiss.normalize_L2(embeddings)
        faiss.normalize_L2(embeddings_labeled)

        index = _initialize_index(embeddings_labeled,
                                  embeddings)

        scores = _get_kl_scores(index,
                                indices_subsampled,
                                embeddings,
                                y_pred_proba,
                                y_pred_proba_labeled,
                                mini_batch_size=self.mini_batch_size,
                                knn_k=self.knn_k,
                                divergence_beta=self.divergence_beta)

        scores = (1 - self.previous_round_weight) * previous_scores + self.previous_round_weight * (1 - scores)
        scores = scores.max() - scores
        scores /= scores.sum()

        indices_selected = np.argpartition(-scores, self.num_pseudo_labels)[:self.num_pseudo_labels]

        return indices_subsampled[indices_selected], y_pred_proba[indices_selected], scores
