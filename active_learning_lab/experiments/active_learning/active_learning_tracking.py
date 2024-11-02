import logging
import random

import pickle
import torch

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
from small_text.data.sampling import _get_class_histogram
from small_text.utils.labels import csr_to_list

from active_learning_lab.utils.metrics import expected_calibration_error


METRIC_COLUMNS = ['train_acc', 'train_micro_precision', 'train_micro_recall', 'train_micro_f1',
                  'train_macro_precision', 'train_macro_recall', 'train_macro_f1', 'train_ece_10',
                  'train_balanced_acc',
                  'test_acc', 'test_micro_precision', 'test_micro_recall', 'test_micro_f1',
                  'test_macro_precision', 'test_macro_recall', 'test_macro_f1', 'test_ece_10',
                  'test_balanced_acc',
                  'valid_acc', 'valid_micro_precision', 'valid_micro_recall', 'valid_micro_f1',
                  'valid_macro_precision', 'valid_macro_recall', 'valid_macro_f1', 'valid_ece_10',
                  'valid_balanced_acc'
                  ]

COLUMNS = ['run_id', 'query_id', 'num_samples', 'query_time_sec', 'update_time_sec'] + \
          METRIC_COLUMNS


def _compute_metrics(y_true, y_pred, y_pred_probas, num_classes):
    multi_label = isinstance(y_true, csr_matrix)

    if y_pred.shape[0] == 0:
        return [MetricsTracker.NO_VALUE] * 8
    else:
        y_pred_probas = np.amax(y_pred_probas, axis=1)

        if multi_label:
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            lb.fit(list(range(num_classes)))
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


class ExperimentTracker(object):

    def __init__(self, num_classes):
        self.query_tracker = QueryTracker()
        self.metrics_tracker = MetricsTracker(num_classes)
        self.run_id = 1

        self.summary_writer = None
        self.stopping_criteria_results = None

    def next_run(self):
        self.query_tracker.reset()
        self.run_id += 1

    def track_initialization_results(self, x_ind_initial, y_initial, run_results):
        self.query_tracker.track_initial_indices(x_ind_initial, y_initial)
        self.metrics_tracker.track(self.run_id, 0, x_ind_initial.shape[0], run_results, self.summary_writer)

        if self.summary_writer is not None:
            acc = self.metrics_tracker.measured_metrics.tail(n=1)['test_acc'].item()
            self.summary_writer.add_scalar(f'Accuracy/Test', acc, global_step=0)

    def track_query(self, indices, labels, scores, run_results, query_id, labeled_pool_size):
        self.query_tracker.track_queried_indices(indices, labels, scores)
        self.metrics_tracker.track(self.run_id,
                                   query_id,
                                   labeled_pool_size,
                                   run_results,
                                   self.summary_writer)

        if self.summary_writer is not None:
            acc = self.metrics_tracker.measured_metrics.tail(n=1)['test_acc'].item()
            self.summary_writer.add_scalar(f'Accuracy/Test', acc, global_step=query_id)

    def track_train_history(self):
        pass

    def track_labeled_pool(self, y_labeled_pool, num_classes, query_id):
        pass

    def track_predict_proba(self, query_id, y_proba_test):
        # round down to the nearest multiple of 0.05 and multiply by 100
        y_proba_test_binned = ((y_proba_test.max(axis=1) // 0.05) * 0.05 * 100)
        self.summary_writer.add_histogram(f'Predictions/Test/Run{self.run_id}', y_proba_test_binned,
                                          global_step=query_id)

    def track_predicted_labels(self, query_id, num_classes, y_train_pred, y_test_pred,
                               y_train_labeled_pred):

        if not isinstance(y_train_pred, csr_matrix):
            if self.summary_writer is not None:
                self.summary_writer.add_histogram(f'Labels/Test/Run{self.run_id}', y_test_pred,
                                             global_step=query_id, bins=list(range(num_classes)))
                self.summary_writer.add_histogram(f'Labels/Train/Run{self.run_id}', y_train_pred,
                                             global_step=query_id, bins=list(range(num_classes)))
                if y_train_labeled_pred.shape[0] > 0:
                    self.summary_writer.add_histogram(f'Labels/TrainL/Run{self.run_id}', y_train_labeled_pred,
                                                 global_step=query_id, bins=list(range(num_classes)))

    def track_query_scores(self, query_id, query_strategy, num_classes, scores,
                           y_train_pred):
        pass

    def track_random_states(self, exp_dir):

        output_path = exp_dir.joinpath(f'random_states.pkl')
        random_states = {
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(legacy=True),
            'torch_random_state': torch.get_rng_state()
        }
        with open(output_path, 'wb+') as f:
            pickle.dump(random_states, f)

        return output_path

    def track_stopping_criteria(self, query_id, stopping_results):

        if self.stopping_criteria_results is None:
            names = [stopping_result[0] for stopping_result in stopping_results]
            stopping_decision_keys = [name + '_stop' for name in names]
            score_keys = [name + '_score' for name in names]

            keys = stopping_decision_keys * 2
            keys[0::2] = stopping_decision_keys
            keys[1::2] = score_keys

            columns = ['run_id', 'query_id'] + keys
            self.stopping_criteria_results = self.measured_metrics = pd.DataFrame(columns=columns)

        stopping_decisions = [stopping_result[1] for stopping_result in stopping_results]
        scores = [stopping_result[2] for stopping_result in stopping_results]

        row = stopping_decisions * 2
        row[0::2] = stopping_decisions
        row[1::2] = scores

        self.stopping_criteria_results.loc[len(self.stopping_criteria_results)] = [
            self.run_id, query_id
        ] + row

    def write_stopping_criteria_results(self, output_file):
        self.stopping_criteria_results.to_csv(output_file, index=False, header=True, na_rep='-')
        return output_file


class MetricsTracker(object):
    NO_VALUE = -1

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.measured_metrics = pd.DataFrame(columns=COLUMNS)

    def track(self, run_id, query_id, num_labeled, run_results, summary_writer):

        times = [run_results.query_time, run_results.update_time]
        metrics_train = _compute_metrics(run_results.y_train_true,
                                         run_results.y_train_pred,
                                         run_results.y_train_pred_proba,
                                         self.num_classes)
        metrics_train_labeled = _compute_metrics(run_results.y_train_subset_true,
                                                 run_results.y_train_subset_pred,
                                                 run_results.y_train_subset_pred_proba,
                                                 self.num_classes)
        metrics_test = _compute_metrics(run_results.y_test_true,
                                        run_results.y_test_pred,
                                        run_results.y_test_pred_proba,
                                        self.num_classes)
        metrics_valid = _compute_metrics(run_results.y_valid_true,
                                         run_results.y_valid_pred,
                                         run_results.y_valid_pred_proba,
                                         self.num_classes)

        logging.info(f'\tTest  Acc: {metrics_test[0] * 100:>4.1f}\t'
                     f'TrainL  Acc: {metrics_train_labeled[0] * 100:>4.1f}\t'
                     f'Train  Acc: {metrics_train[0] * 100:>4.1f}\t')
        logging.info(f'\tTest BAcc: {metrics_test[-1] * 100:>4.1f}\t'
                     f'TrainL BAcc: {metrics_train_labeled[-1] * 100:>4.1f}\t'
                     f'Train BAcc: {metrics_train[-1] * 100:>4.1f}\t')
        logging.info(f'\tTest  mF1: {metrics_test[3] * 100:>4.1f}\t'
                     f'TrainL  mF1: {metrics_train_labeled[3] * 100:>4.1f}\t'
                     f'Train  mF1: {metrics_train[3] * 100:>4.1f}\t')
        logging.info(f'\tTest  MF1: {metrics_test[-3] * 100:>4.1f}\t'
                     f'TrainL  MF1: {metrics_train_labeled[-3] * 100:>4.1f}\t'
                     f'Train  MF1: {metrics_train[-3] * 100:>4.1f}\t')
        logging.info(f'\tTest  ECE: {metrics_test[-2] * 100:>4.1f}\t'
                     f'TrainL  ECE: {metrics_train_labeled[-2] * 100:>4.1f}\t'
                     f'Train  ECE: {metrics_train[-2] * 100:>4.1f}')
        logging.info('')

        report = classification_report(run_results.y_test_true, run_results.y_test_pred, output_dict=True)
        results_per_label = [f'{label}={report[str(label)]["f1-score"]:0.2f}' for label in range(self.num_classes)]
        logging.info('\tTest F1 per class: ' + ','.join(results_per_label))

        report = classification_report(run_results.y_train_true, run_results.y_train_pred, output_dict=True)
        results_per_label = [f'{label}={report[str(label)]["f1-score"]:0.2f}' if str(label) in report else f'{label}=--'
                             for label in range(self.num_classes)]
        logging.info('\tTrain F1 per class: ' + ','.join(results_per_label))
        logging.info('')

        summary_writer.add_scalar(f'ECE/TrainL/Run{run_id}', metrics_test[-2], query_id)
        summary_writer.add_scalar(f'ECE/Test/Run{run_id}', metrics_train_labeled[-2], query_id)
        summary_writer.add_scalar(f'ECE/Train/Run{run_id}', metrics_train[-2], query_id)

        self.measured_metrics.loc[len(self.measured_metrics)] = [int(run_id), int(query_id),
                                                                 num_labeled] + \
                                                                times + metrics_train + metrics_test + metrics_valid

    def track_train_history(self, query_id, train_history_dir, train_history):
        data = []

        selected_model = train_history.selected_model
        for i, entry in enumerate(train_history.metric_history):
            train_loss, train_acc, valid_loss, valid_acc = entry
            data.append([train_loss, train_acc, valid_loss, valid_acc, selected_model == i])

        df = pd.DataFrame(data, columns=['train_loss', 'train_acc', 'valid_loss', 'valid_acc',
                                         'is_selected_model'])

        output_file = Path(train_history_dir).joinpath(str(query_id) + '.csv')
        df.to_csv(output_file, index=False, header=True)

    def write(self, output_file):
        self.measured_metrics = self.measured_metrics \
            .astype({'run_id': int, 'query_id': int, 'num_samples': int})
        self.measured_metrics.to_csv(output_file, index=False, header=True)

        return output_file

    def write_aggregate(self, output_file):
        df = self.measured_metrics.groupby(['query_id'])[METRIC_COLUMNS].agg(lambda x: [np.mean(x), np.std(x, ddof=0)])
        df.columns = df.columns.to_flat_index()
        df.columns = [tup[0] + '_' + tup[1] for tup in df.columns]
        df.to_csv(output_file, index=False, header=True)

        return output_file


class QueryTracker(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.query_data = {
            'initial_indices': None,
            'initial_labels': [],
            'queried_indices': [],
            'queried_labels': [],
            'queried_scores': []
        }

    def track_initial_indices(self, indices, labels):
        if self.query_data['initial_indices'] is None:
            self.query_data['initial_indices'] = indices.tolist()
            if isinstance(labels, csr_matrix):
                self.query_data['initial_labels'] = csr_to_list(labels)
            else:
                self.query_data['initial_labels'] = labels.tolist()
        else:
            raise ValueError('Initial indices can only bet set once')

    def track_queried_indices(self, indices, labels, scores):

        self.query_data['queried_indices'].append(indices.tolist())
        if isinstance(labels, csr_matrix):
            self.query_data['queried_labels'].append(csr_to_list(labels))
        else:
            self.query_data['queried_labels'].append(labels.tolist())

        if scores is None:
            self.query_data['queried_scores'].append(scores)
        else:
            self.query_data['queried_scores'].append(scores.tolist())

    def write(self, output_path):

        if len(self.query_data['queried_scores']) > 0 and \
                all([score is None for score in self.query_data['queried_scores']]):
            self.query_data['queried_scores'] = None

        np.savez(output_path,
                 initial_indices=self.query_data['initial_indices'],
                 initial_labels=self.query_data['initial_labels'],
                 queried_indices=self.query_data['queried_indices'],
                 queried_labels=self.query_data['queried_labels'],
                 queried_scores=self.query_data['queried_scores'])

        return output_path


class RunResults(object):
    """
    Stores the results of a single run.
    """

    def __init__(self,
                 query_time: int,
                 update_time: int,
                 y_train_true,
                 y_train_pred,
                 y_train_pred_proba,
                 y_train_subset_true,
                 y_train_subset_pred,
                 y_train_subset_pred_proba,
                 y_valid_true,
                 y_valid_pred,
                 y_valid_pred_proba,
                 y_test_true,
                 y_test_pred,
                 y_test_pred_proba,
                 additional_results: dict=dict()):
        """
        Parameters
        ----------
        query_time : int

        update_time : int

        y_train_true : ndarray (int)

        """
        self.query_time = query_time
        self.update_time = update_time
        self.y_train_true = y_train_true
        self.y_train_pred = y_train_pred
        self.y_train_pred_proba = y_train_pred_proba
        self.y_train_subset_true = y_train_subset_true
        self.y_train_subset_pred = y_train_subset_pred
        self.y_train_subset_pred_proba = y_train_subset_pred_proba
        self.y_valid_true = y_valid_true
        self.y_valid_pred = y_valid_pred
        self.y_valid_pred_proba = y_valid_pred_proba
        self.y_test_true = y_test_true
        self.y_test_pred = y_test_pred
        self.y_test_pred_proba = y_test_pred_proba
        self.additional_results = additional_results
