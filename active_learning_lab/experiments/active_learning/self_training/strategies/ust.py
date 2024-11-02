import logging

import numpy as np

from dependency_injector.wiring import inject, Provide

from small_text.data.sampling import _get_class_histogram
from small_text.query_strategies.bayesian import _bald
from torch.utils.tensorboard import SummaryWriter

from active_learning_lab.utils.pytorch import free_resources_fix
from active_learning_lab.experiments.active_learning.self_training.strategies.base import (
    SelfTrainingStrategy,
    SelfTrainingStatus,
    SelfTrainingOverallResults
)
from active_learning_lab.experiments.active_learning.self_training.utils.class_balance import _query_class_balanced
from active_learning_lab.experiments.active_learning.self_training.utils.ust import change_dropout_rate
from active_learning_lab.utils.time import measure_time_context


logger = logging.getLogger(__name__)


class UST(SelfTrainingStrategy):

    @inject
    def __init__(self,
                 sampling_strategy='easy',
                 self_training_iterations=1,
                 dropout_sampling=10,
                 k=30,
                 alpha=0.1,
                 subsample_size=16384,
                 mini_batch_size=32,
                 summary_writer: SummaryWriter = Provide['summary_writer']):

        self.self_training_iterations = self_training_iterations

        if sampling_strategy not in ['easy', 'hard']:
            raise ValueError(f'Invalid sampling_strategy: {sampling_strategy}')

        self.sampling_strategy = sampling_strategy
        self.dropout_sampling = dropout_sampling
        self.k = k
        self.alpha = alpha

        self.subsample_size = subsample_size
        self.mini_batch_size = mini_batch_size

        self.summary_writer = summary_writer

        self.indices_labeled = set()
        self.index = None

    def train(self, clf, dataset, y, indices_unlabeled, indices_labeled, indices_valid, test_set=None):

        y_train_pred_proba_before = clf.predict_proba(dataset)
        y_test_pred_proba_before = clf.predict_proba(test_set)

        indices_pseudo_labeled = np.copy(indices_labeled)
        y_pseudo_labeled = np.copy(y)

        results = []

        for t in range(self.self_training_iterations):

            result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, y_train_pred_proba, y_test_pred_proba = \
                self._self_training_iteration(
                    clf,
                    dataset,
                    indices_labeled,
                    indices_pseudo_labeled,
                    y_pseudo_labeled,
                    indices_unlabeled,
                    indices_valid,
                    test_set
                )
            results.append(result)
            free_resources_fix()

            indices_unlabeled = indices_unlabeled
            indices_pseudo_labeled = indices_pseudo_labeled

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

    def _self_training_iteration(self, clf, dataset, indices_labeled, indices_pseudo_labeled, y_pseudo_labeled,
                                 indices_unlabeled, indices_valid, test_set):

        if self.subsample_size is None:
            indices_subsampled = indices_unlabeled
        else:
            indices_subsampled = np.random.choice(indices_unlabeled,
                                                  min(self.subsample_size, indices_unlabeled.shape[0]),
                                                  replace=False)

        with measure_time_context() as pseudo_label_query_time:
            y_pred_proba = clf.predict_proba(dataset[indices_subsampled], dropout_sampling=self.dropout_sampling)
            y_pred = y_pred_proba.mean(axis=1).argmax(axis=1)

            y_pred_proba_labeled = clf.predict_proba(dataset[indices_labeled], dropout_sampling=self.dropout_sampling)
            y_pred_labeled = y_pred_proba_labeled.mean(axis=1).argmax(axis=1)

            if self.sampling_strategy == 'easy':
                scores = 1 - _bald(y_pred_proba)
            else:
                scores = _bald(y_pred_proba)
            scores = np.maximum(np.zeros(scores.shape), scores)

            y = dataset.y[indices_pseudo_labeled]
            indices_new = _query_class_balanced(y_pred, scores, clf.num_classes, y, self.k * clf.num_classes)

        indices_pseudo_labeled = np.append(indices_pseudo_labeled, indices_new)
        y_pseudo_labeled = np.append(y_pseudo_labeled, y_pred[indices_new])

        indices_unlabeled = set(indices_unlabeled.tolist())
        indices_unlabeled = indices_unlabeled - set(indices_new.tolist())
        indices_unlabeled = np.array(list(indices_unlabeled))

        dataset_new = dataset[indices_pseudo_labeled].clone()
        dataset_new.y = y_pseudo_labeled

        validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]
        with measure_time_context() as update_time:
            weights_labeled = self._get_weights(clf, y_pred_labeled, y_pred_proba_labeled, np.s_[:])

            # TODO: does this work for multiple self-training iterations?
            weights_pseudo_labels = self._get_weights(clf, y_pred, y_pred_proba, indices_new)
            weights = np.append(weights_labeled, weights_pseudo_labels)

            change_dropout_rate(clf.model, dropout=0.5, attention_dropout=0.3, hidden_dropout=0.3)
            self._train(clf,
                        dataset_new,
                        validation_set,
                        indices_labeled.shape[0],
                        indices_pseudo_labeled.shape[0] - indices_labeled.shape[0],
                        train_weights=weights)
            change_dropout_rate(clf.model)

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

        logger.info(f'[self-training-strategy] Self-Training label distribution '
                    f'{_get_class_histogram(dataset.y[indices_pseudo_labeled], clf.num_classes)} .')

        return result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, y_train_pred_proba, y_test_pred_proba

    def _get_weights(self, clf, y_pred, y_pred_proba, indices_target):
        y_target = y_pred[indices_target]
        cond = [
            [
                [y == c for c in range(clf.num_classes)]
                for y in y_target
                for _ in range(self.dropout_sampling)
            ]
        ]
        weights_ = np.log(1 / (np.var(np.extract(cond, y_pred_proba[indices_target]).reshape(y_target.shape[0], -1),
                                      axis=1) + 1e-8)) * self.alpha
        weights_ = np.maximum(weights_, self.alpha / 0.25)  # var of a variable in [0, 1] is lte 0.25

        return weights_
