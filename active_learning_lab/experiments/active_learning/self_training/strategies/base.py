import logging
import numpy as np

from enum import Enum
from math import ceil
from scipy.special import rel_entr

from small_text import SetFitClassification
from small_text.data.sampling import _get_class_histogram

from active_learning_lab.utils.time import measure_time_context

from active_learning_lab.experiments.active_learning.self_training.classification.setfit import SetFitClassificationExtended
from sentence_transformers import losses
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction


logger = logging.getLogger(__name__)


class SelfTrainingOverallResults(object):
    """
    Stores the results of a single self-training run.
    """
    def __init__(self,
                 iterations,
                 num_samples,
                 num_pseudo_labels,
                 y_train_true,
                 y_train_pred_before,
                 y_train_pred_proba_before,
                 y_train_pred_after,
                 y_train_pred_proba_after,
                 y_test_true,
                 y_test_pred_before,
                 y_test_pred_proba_before,
                 y_test_pred_after,
                 y_test_pred_proba_after):
        """
        Parameters
        ----------

        """
        self.iterations = iterations
        self.num_samples = num_samples
        self.num_pseudo_labels = num_pseudo_labels

        self.y_train_true = y_train_true
        self.y_train_pred_before = y_train_pred_before
        self.y_train_pred_proba_before = y_train_pred_proba_before
        self.y_train_pred_after = y_train_pred_after
        self.y_train_pred_proba_after = y_train_pred_proba_after

        self.y_test_true = y_test_true
        self.y_test_pred_before = y_test_pred_before
        self.y_test_pred_proba_before = y_test_pred_proba_before
        self.y_test_pred_after = y_test_pred_after
        self.y_test_pred_proba_after = y_test_pred_proba_after


class SelfTrainingStatus(Enum):
    SUCCESS = 'success',
    FAILURE_ONE_CLASS = 'failure-just-one-class',
    FAILURE_NO_PSEUDO_LABELS_AFTER_THRESHOLDING = 'failure-no-pseudo-labels-after-thresholding',
    # verips
    FAILURE_VERIPS_NO_PSEUDO_LABELS_AFTER_VERIFICATION = 'failure-verips-no-pseudo-labels-verification',


class SelfTrainingIterationRunResults(object):
    """
    Stores the results of a single run's iteration.
    """

    def __init__(self,
                 num_samples,
                 num_pseudo_labels,
                 status: SelfTrainingStatus,
                 pseudo_label_query_time: int,
                 update_time: int,
                 evaluation_time: int,
                 y_pseudo_labels_new_true,
                 y_pseudo_labels_new_pred,
                 y_pseudo_labels_new_pred_proba,
                 y_pseudo_labels_new_imbalance_ratio,
                 y_pseudo_labels_new_imbalance_kldiv,
                 y_pseudo_labels_all_true,
                 y_pseudo_labels_all_pred,
                 y_pseudo_labels_all_pred_proba,
                 y_pseudo_labels_all_imbalance_ratio,
                 y_pseudo_labels_all_imbalance_kldiv,
                 y_train_true,
                 y_train_pred,
                 y_train_pred_proba,
                 y_test_true,
                 y_test_pred,
                 y_test_pred_proba):
        """
        Parameters
        ----------
        pseudo_label_query_time : int

        update_time : int

        y_train_true : ndarray (int)

        """
        self.num_samples = num_samples
        self.num_pseudo_labels = num_pseudo_labels
        self.status = status
        self.pseudo_label_query_time = pseudo_label_query_time
        self.update_time = update_time
        self.evaluation_time = evaluation_time
        self.y_pseudo_labels_new_true = y_pseudo_labels_new_true
        self.y_pseudo_labels_new_pred = y_pseudo_labels_new_pred
        self.y_pseudo_labels_new_pred_proba = y_pseudo_labels_new_pred_proba
        self.y_pseudo_labels_new_imbalance_ratio = y_pseudo_labels_new_imbalance_ratio
        self.y_pseudo_labels_new_imbalance_kldiv = y_pseudo_labels_new_imbalance_kldiv
        self.y_pseudo_labels_all_true = y_pseudo_labels_all_true
        self.y_pseudo_labels_all_pred = y_pseudo_labels_all_pred
        self.y_pseudo_labels_all_pred_proba = y_pseudo_labels_all_pred_proba
        self.y_pseudo_labels_all_imbalance_ratio = y_pseudo_labels_all_imbalance_ratio
        self.y_pseudo_labels_all_imbalance_kldiv = y_pseudo_labels_all_imbalance_kldiv
        self.y_train_true = y_train_true
        self.y_train_pred = y_train_pred
        self.y_train_pred_proba = y_train_pred_proba
        self.y_test_true = y_test_true
        self.y_test_pred = y_test_pred
        self.y_test_pred_proba = y_test_pred_proba


class SelfTrainingStrategy(object):

    def __init__(self, reuse_model=True):
        self.reuse_model = reuse_model

    def _abort(self, clf, dataset, test_set, validation_set, indices_labeled, indices_unlabeled,
               indices_new, indices_pseudo_labeled, status, pseudo_label_query_time,
               partial_update_time=0, train_weights=None):

        if hasattr(self, 'reuse_model') and self.reuse_model:
            with measure_time_context() as update_time_two:
                self._train(clf,
                            dataset[indices_pseudo_labeled].clone(),
                            validation_set,
                            indices_labeled.shape[0],
                            indices_pseudo_labeled.shape[0] - indices_labeled.shape[0],
                            train_weights=train_weights)
        else:
            def update_time_two():
                return 0

        result, y_train_pred_proba, y_test_pred_proba = self._evaluate(
            clf,
            dataset,
            test_set,
            indices_labeled,
            indices_pseudo_labeled,
            indices_new,
            status,
            pseudo_label_query_time,
            partial_update_time + update_time_two()
        )

        logger.info(f'[self-training-strategy] Self-Training label distribution '
                    f'{_get_class_histogram(dataset.y[indices_pseudo_labeled], clf.num_classes)} .')

        return result, indices_unlabeled, indices_pseudo_labeled, dataset[indices_pseudo_labeled].y, \
               y_train_pred_proba, y_test_pred_proba

    def _evaluate(self, clf, dataset, test_set, indices_labeled, indices_pseudo_labeled, indices_new, status,
                  pseudo_label_query_time, update_time):

        with measure_time_context() as evaluation_time:
            y_train_pred_proba = clf.predict_proba(dataset)
            y_train_pred = np.argmax(y_train_pred_proba, 1)
            y_test_pred_proba = clf.predict_proba(test_set)

        result = get_iteration_results(clf.num_classes,
                                       indices_labeled.shape[0],
                                       indices_pseudo_labeled.shape[0],
                                       status,
                                       pseudo_label_query_time,
                                       update_time,
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

        logger.info(f'[self-training-strategy] Self-Training label distribution '
                    f'{_get_class_histogram(dataset.y[indices_pseudo_labeled], clf.num_classes)} .')

        return result, y_train_pred_proba, y_test_pred_proba

    def _train(self, clf, train_set, validation_set, num_labeled, num_pseudo_labeled,
               num_iterations=5, train_weights=None, train_key='train'):

        if train_weights is not None:
            self.self_training_tracking.track_weights(train_key, train_weights)

        from unittest.mock import patch
        from small_text.integrations.transformers.classifiers.setfit import SetFitTrainer

        current_num_examples = num_labeled * num_iterations * 2

        num_iterations_new = int(ceil(current_num_examples / (num_labeled + num_pseudo_labeled) / 2))
        num_iterations_new = max(num_iterations_new, 1)
        logger.info(f'[self-training-strategy] num_iterations_new={num_iterations_new}')

        # <newinit>
        def init_new(
                self,
                model=None,
                train_dataset=None,
                eval_dataset=None,
                model_init=None,
                metric="accuracy",
                loss_class=losses.CosineSimilarityLoss,
                num_iterations: int = num_iterations_new,
                num_epochs: int = 1,
                learning_rate: float = 2e-5,
                batch_size: int = 16,
                seed: int = 42,
                column_mapping=None,
                use_amp: bool = False,
                warmup_proportion: float = 0.1,
                distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
                margin: float = 0.25,
                samples_per_label: int = 3,
        ):

            if (warmup_proportion < 0.0) or (warmup_proportion > 1.0):
                raise ValueError(
                    f"warmup_proportion must be greater than or equal to 0.0 and less than or equal to 1.0! But it was: {warmup_proportion}"
                )

            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.model_init = model_init
            self.metric = metric
            self.loss_class = loss_class
            self.num_iterations = num_iterations
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.seed = seed
            self.column_mapping = column_mapping
            self.use_amp = use_amp
            self.warmup_proportion = warmup_proportion
            self.distance_metric = distance_metric
            self.margin = margin
            self.samples_per_label = samples_per_label

            if model is None:
                if model_init is not None:
                    model = self.call_model_init()
                else:
                    raise RuntimeError("`SetFitTrainer` requires either a `model` or `model_init` argument")
            else:
                if model_init is not None:
                    raise RuntimeError(
                        "`SetFitTrainer` requires either a `model` or `model_init` argument, but not both")

            self.model = model
            self.hp_search_backend = None
            self._freeze = True  # If True, will train the body only; otherwise, train the body and head
        # </end newinit>

        if isinstance(clf, (SetFitClassification, SetFitClassificationExtended)):
            with patch.object(SetFitTrainer, '__init__', init_new):
                clf.fit(train_set, validation_set=validation_set, weights=train_weights)
        else:
            clf.fit(train_set, validation_set=validation_set, weights=train_weights)


def get_iteration_results(num_classes,
                          num_labels,
                          num_pseudo_labels,
                          status,
                          pseudo_label_query_time,
                          update_time,
                          evaluation_time,
                          y_pseudo_labels_new_true,
                          y_pseudo_labels_new_pred,
                          y_pseudo_labels_new_proba,
                          y_pseudo_labels_all_true,
                          y_pseudo_labels_all_pred,
                          y_pseudo_labels_all_proba,
                          y_train_true,
                          y_train_pred,
                          y_train_pred_proba,
                          y_test_true,
                          y_test_pred,
                          y_test_pred_proba):

    y_pseudo_labels_new_hist = _get_class_histogram(y_pseudo_labels_new_pred, num_classes)
    if y_pseudo_labels_new_hist.sum() > 0:
        y_pseudo_labels_new_hist_rel = y_pseudo_labels_new_hist / y_pseudo_labels_new_hist.sum()
    else:
        y_pseudo_labels_new_hist_rel = y_pseudo_labels_new_hist.astype(float)

    y_pseudo_labels_all_hist = _get_class_histogram(y_pseudo_labels_all_pred, num_classes)
    if y_pseudo_labels_all_hist.sum() > 0:
        y_pseudo_labels_all_hist_rel = y_pseudo_labels_all_hist / y_pseudo_labels_all_hist.sum()
    else:
        y_pseudo_labels_all_hist_rel = y_pseudo_labels_all_hist.astype(float)

    hist_rel = np.ones_like(y_pseudo_labels_new_hist_rel) / num_classes

    return SelfTrainingIterationRunResults(num_labels,
                                           num_pseudo_labels,
                                           status,
                                           pseudo_label_query_time,
                                           update_time,
                                           evaluation_time,
                                           y_pseudo_labels_new_true,
                                           y_pseudo_labels_new_pred,
                                           y_pseudo_labels_new_proba,
                                           (y_pseudo_labels_new_hist.max()+1) / (y_pseudo_labels_new_hist.min()+1),
                                           rel_entr(y_pseudo_labels_new_hist_rel, hist_rel).sum(),
                                           y_pseudo_labels_all_true,
                                           y_pseudo_labels_all_pred,
                                           y_pseudo_labels_all_proba,
                                           (y_pseudo_labels_all_hist.max()+1) / (y_pseudo_labels_all_hist.min()+1),
                                           rel_entr(y_pseudo_labels_all_hist_rel, hist_rel).sum(),
                                           y_train_true,
                                           y_train_pred,
                                           y_train_pred_proba,
                                           y_test_true,
                                           y_test_pred,
                                           y_test_pred_proba)


def get_empty_iteration_result(num_classes,
                               num_labels,
                               status,
                               pseudo_label_query_time,
                               update_time=0):

    return SelfTrainingIterationRunResults(num_labels,
                                           0,
                                           status,
                                           pseudo_label_query_time,
                                           update_time,
                                           0,
                                           np.empty((0, num_classes), dtype=int),
                                           np.empty((0,), dtype=int),
                                           np.empty((0, num_classes), dtype=float),
                                           -np.inf,
                                           -np.inf,
                                           np.empty([], dtype=int),
                                           np.empty([], dtype=int),
                                           np.empty((0, num_classes), dtype=float),
                                           -np.inf,
                                           -np.inf,
                                           np.empty((0, num_classes), dtype=int),
                                           np.empty((0,), dtype=int),
                                           np.empty((0, num_classes), dtype=float),
                                           np.empty((0, num_classes), dtype=int),
                                           np.empty((0,), dtype=int),
                                           np.empty((0, num_classes), dtype=float))
