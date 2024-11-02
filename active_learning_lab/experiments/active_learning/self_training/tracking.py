import numpy as np
import pandas as pd

from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from active_learning_lab.experiments.active_learning.self_training.strategies.base import (
    SelfTrainingOverallResults,
    SelfTrainingIterationRunResults
)
from active_learning_lab.experiments.active_learning.active_learning_tracking import MetricsTracker, METRIC_COLUMNS
from active_learning_lab.utils.metrics import expected_calibration_error


COLUMNS_OVERALL = [
    'run_id', 'query_id', 'num_samples', 'num_pseudo_labels',
    'train_acc_before', 'train_micro_precision_before', 'train_micro_recall_before', 'train_micro_f1_before',
    'train_macro_precision_before', 'train_macro_recall_before', 'train_macro_f1_before', 'train_ece_10_before',
    'train_balanced_acc_before',
    'train_acc_after', 'train_micro_precision_after', 'train_micro_recall_after', 'train_micro_f1_after',
    'train_macro_precision_after', 'train_macro_recall_after', 'train_macro_f1_after', 'train_ece_10_after',
    'train_balanced_acc_after',
    'test_acc_before', 'test_precision_before', 'test_micro_recall_before', 'test_micro_f1_before',
    'test_macro_precision_before', 'test_macro_recall_before', 'test_macro_f1_before', 'test_ece_10_before',
    'test_balanced_acc_before',
    'test_acc_after', 'test_micro_precision_after', 'test_micro_recall_after', 'test_micro_f1_after',
    'test_macro_precision_after', 'test_macro_recall_after', 'test_macro_f1_after', 'test_ece_10_after',
    'test_balanced_acc_after',
]


PL_METRIC_COLUMNS = [
    'pseudo_labels_new_acc', 'pseudo_labels_new_micro_precision', 'pseudo_labels_new_micro_recall', 'pseudo_labels_new_micro_f1',
    'pseudo_labels_new_macro_precision', 'pseudo_labels_new_macro_recall', 'pseudo_labels_new_macro_f1', 'pseudo_labels_new_ece_10', 'pseudo_labels_new_balanced_acc',
    'pseudo_labels_new_imbalance_ratio', 'pseudo_labels_new_imbalance_kldiv',
    'pseudo_labels_acc', 'pseudo_labels_micro_precision', 'pseudo_labels_micro_recall', 'pseudo_labels_micro_f1',
    'pseudo_labels_macro_precision', 'pseudo_labels_macro_recall', 'pseudo_labels_macro_f1', 'pseudo_labels_ece_10', 'pseudo_labels_balanced_acc',
    'pseudo_labels_imbalance_ratio', 'pseudo_labels_imbalance_kldiv',
]


COLUMNS_DETAILED = [
    'run_id', 'query_id', 'num_samples', 'num_pseudo_labels', 'status', 'query_time_sec', 'update_time_sec',
    'evaluation_time_sec'
] + PL_METRIC_COLUMNS + METRIC_COLUMNS[:18]


class SelfTrainingExperimentTracker(object):

    def __init__(self, run_id: int, num_classes: int):
        self.run_id = run_id

        self.num_classes = num_classes
        self.overall_metrics = pd.DataFrame(columns=COLUMNS_OVERALL)
        self.detailed_metrics = pd.DataFrame(columns=COLUMNS_DETAILED)

        self.weight_keys = Counter()
        self.weights = dict()

    def track(self, query_id: int, results: SelfTrainingOverallResults):
        self._track_overall(query_id, results)
        for iteration_results in results.iterations:
            self._track_detailed(query_id, iteration_results)

    def _track_overall(self, query_id: int, results: SelfTrainingOverallResults):
        metrics = [
            self.run_id, query_id, results.num_samples, results.num_pseudo_labels,
        ] + self._compute_metrics(results.y_train_true, results.y_train_pred_before, results.y_train_pred_proba_before) \
          + self._compute_metrics(results.y_train_true, results.y_train_pred_after, results.y_train_pred_proba_after) \
          + self._compute_metrics(results.y_test_true, results.y_test_pred_before, results.y_test_pred_proba_before) \
          + self._compute_metrics(results.y_test_true, results.y_test_pred_after, results.y_test_pred_proba_after)

        self.overall_metrics.loc[len(self.overall_metrics)] = metrics

    def _track_detailed(self, query_id: int, iteration_results: SelfTrainingIterationRunResults):
        metrics = [
            self.run_id, query_id, iteration_results.num_samples, iteration_results.num_pseudo_labels,
            iteration_results.status, iteration_results.pseudo_label_query_time, iteration_results.update_time,
            iteration_results.evaluation_time
        ] + self._compute_metrics(iteration_results.y_pseudo_labels_new_true,
                                  iteration_results.y_pseudo_labels_new_pred,
                                  iteration_results.y_pseudo_labels_new_pred_proba) \
            + [iteration_results.y_pseudo_labels_new_imbalance_ratio, iteration_results.y_pseudo_labels_new_imbalance_kldiv] \
            + self._compute_metrics(iteration_results.y_pseudo_labels_all_true,
                                    iteration_results.y_pseudo_labels_all_pred,
                                    iteration_results.y_pseudo_labels_all_pred_proba) \
            + [iteration_results.y_pseudo_labels_all_imbalance_ratio, iteration_results.y_pseudo_labels_all_imbalance_kldiv] \
            + self._compute_metrics(iteration_results.y_train_true,
                                    iteration_results.y_train_pred,
                                    iteration_results.y_train_pred_proba) \
            + self._compute_metrics(iteration_results.y_test_true,
                                    iteration_results.y_test_pred,
                                    iteration_results.y_test_pred_proba)

        self.detailed_metrics.loc[len(self.detailed_metrics)] = metrics

    def _compute_metrics(self, y_true, y_pred, y_pred_probas):

        multi_label = isinstance(y_true, csr_matrix)

        if len(y_pred.shape) == 0 or y_pred.shape[0] == 0:
            return [MetricsTracker.NO_VALUE] * 9
        else:
            y_pred_probas = np.amax(y_pred_probas, axis=1)

            if multi_label:
                from sklearn.preprocessing import LabelBinarizer
                lb = LabelBinarizer()
                lb.fit(list(range(self.num_classes)))
                y_true = y_true.toarray()
                y_true = np.apply_along_axis(lambda x: lb.transform(x).flatten(), 1, y_true)
                y_pred = y_pred.toarray()
                y_pred = np.apply_along_axis(lambda x: lb.transform(x).flatten(), 1, y_pred)

            return [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred, average='micro'),
                recall_score(y_true, y_pred, average='micro'),
                f1_score(y_true, y_pred, average='micro'),
                precision_score(y_true, y_pred, average='macro'),
                recall_score(y_true, y_pred, average='macro'),
                f1_score(y_true, y_pred, average='macro'),
                expected_calibration_error(y_pred, y_pred_probas, y_true),
                balanced_accuracy_score(y_true, y_pred) if not multi_label else -np.nan
            ]

    def track_weights(self, key, weights):
        self.weight_keys.update(key=1)
        self.weights[key + '_' + str(self.weight_keys[key])] = weights

    def write(self, self_training_overall_path, self_training_detailed_path, weights_path):
        self.overall_metrics = self.overall_metrics \
            .astype({'run_id': int, 'query_id': int, 'num_samples': int, 'num_pseudo_labels': int})
        self.overall_metrics.to_csv(self_training_overall_path, index=False, header=True)

        self.detailed_metrics = self.detailed_metrics \
            .astype({'run_id': int, 'query_id': int, 'num_samples': int, 'num_pseudo_labels': int})
        self.detailed_metrics.to_csv(self_training_detailed_path, index=False, header=True)

        np.savez(weights_path, **self.weights)
