import logging

import numpy as np

from dependency_injector.wiring import inject, Provide
from torch.utils.tensorboard import SummaryWriter
from small_text.data.sampling import _get_class_histogram

from active_learning_lab.experiments.active_learning.self_training.strategies.base import (
    SelfTrainingStrategy,
    SelfTrainingStatus,
    SelfTrainingOverallResults
)
from active_learning_lab.utils.time import measure_time_context


logger = logging.getLogger(__name__)


class VERIPS(SelfTrainingStrategy):

    @inject
    def __init__(self,
                 scoring_method: str,
                 self_training_iterations=1,
                 lmbda=0.1,
                 subsample_size=16384,
                 mini_batch_size=32,
                 reuse_model=True,
                 summary_writer: SummaryWriter = Provide['summary_writer']):
                 # subsample_size=16384, self_training_size(R)=2048 in (Mukherjee, 2020)
        """
        Parameters
        ----------

        """
        super().__init__(reuse_model=reuse_model)

        if scoring_method not in ['entropy', 'margin']:
            raise ValueError(f'Invalid scoring method: scoring_method={scoring_method}')

        self.scoring_method = scoring_method

        self.self_training_iterations = self_training_iterations

        self.lmbda = lmbda
        self.subsample_size = subsample_size
        self.mini_batch_size = mini_batch_size

        self.summary_writer = summary_writer

        self.indices_labeled = set()
        self.index = None

    def train(self, clf, dataset, y, indices_unlabeled, indices_labeled, indices_valid, test_set=None):
        assert indices_valid is None

        y_train_pred_proba_before = clf.predict_proba(dataset)
        y_test_pred_proba_before = clf.predict_proba(test_set)

        indices_pseudo_labeled = np.copy(indices_labeled)
        y_pseudo_labeled = np.copy(y)

        results = []

        for _ in range(self.self_training_iterations):

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

        with measure_time_context() as pseudo_label_query_time:
            indices_new = self.get_pseudo_labeled_dataset(clf, dataset, indices_unlabeled)

        if indices_new.shape[0] == 0:
            logger.info(f'[self-training-strategy] No pseudo-labels exceeded threshold. '
                        f'No pseudo-labels before verification step.')
            validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]
            return self._abort(clf,
                               dataset,
                               test_set,
                               validation_set,
                               indices_labeled,
                               indices_unlabeled,
                               indices_new,
                               indices_pseudo_labeled,
                               SelfTrainingStatus.FAILURE_NO_PSEUDO_LABELS_AFTER_THRESHOLDING,
                               pseudo_label_query_time())
        else:
            y_pred_verification = clf.predict(dataset[indices_new])

            validation_set = None if indices_valid is None else dataset[indices_labeled[indices_valid]]
            with measure_time_context() as update_verification_model_time:
                clf.fit(dataset[indices_pseudo_labeled], validation_set=validation_set)

            y_pred = clf.predict(dataset[indices_new])
            indices_verified = np.arange(indices_new.shape[0])[y_pred == y_pred_verification]

            if indices_verified.sum() == 0:
                logger.info(f'[self-training-strategy] No pseudo-labels could be verified. '
                            f'No self-training after verification step.')
                return self._abort(clf,
                                   dataset,
                                   test_set,
                                   validation_set,
                                   indices_labeled,
                                   indices_unlabeled,
                                   indices_new,
                                   indices_pseudo_labeled,
                                   SelfTrainingStatus.FAILURE_VERIPS_NO_PSEUDO_LABELS_AFTER_VERIFICATION,
                                   pseudo_label_query_time(),
                                   partial_update_time=update_verification_model_time())

            indices_pseudo_labeled = np.append(indices_pseudo_labeled, indices_new[indices_verified])
            y_pseudo_labeled = np.append(y_pseudo_labeled, y_pred[indices_verified])

            indices_unlabeled = set(indices_unlabeled.tolist())
            indices_unlabeled = indices_unlabeled - set(indices_new[indices_verified].tolist())
            indices_unlabeled = np.array(list(indices_unlabeled))

            dataset_new = dataset[indices_pseudo_labeled].clone()
            dataset_new.y = y_pseudo_labeled

            with measure_time_context() as update_verification_model_time_two:
                self._train(clf,
                            dataset_new,
                            validation_set,
                            indices_labeled.shape[0],
                            indices_pseudo_labeled.shape[0] - indices_labeled.shape[0])

            result, y_train_pred_proba, y_test_pred_proba = self._evaluate(
                clf,
                dataset,
                test_set,
                indices_labeled,
                indices_pseudo_labeled,
                indices_new,
                SelfTrainingStatus.SUCCESS,
                pseudo_label_query_time(),
                update_verification_model_time() + update_verification_model_time_two()
            )

            logger.info(f'[self-training-strategy] Self-Training label distribution '
                        f'{_get_class_histogram(dataset.y[indices_pseudo_labeled], clf.num_classes)} .')

            return result, indices_unlabeled, indices_pseudo_labeled, y_pseudo_labeled, \
                   y_train_pred_proba, y_test_pred_proba

    def get_pseudo_labeled_dataset(self, clf, dataset, indices_unlabeled):

        if self.subsample_size is None:
            indices_subsampled = indices_unlabeled
        else:
            indices_subsampled = np.random.choice(indices_unlabeled,
                                                  min(self.subsample_size, indices_unlabeled.shape[0]),
                                                  replace=False)

        y_pred, y_pred_proba = clf.predict(dataset[indices_subsampled], return_proba=True)

        if self.scoring_method == 'entropy':
            from scipy.stats import entropy
            # 1 - normalized entropy (higher is better)
            score = 1 - np.apply_along_axis(lambda x: entropy(x) / np.log(clf.num_classes), 1, y_pred_proba)
        elif self.scoring_method == 'margin':
            ind = np.argsort(y_pred_proba)
            # 1 - margin (higher is better)
            score = y_pred_proba[ind[-1]] - y_pred_proba[ind[-2]]
        else:
            raise NotImplementedError

        indices_selected = np.argwhere(score >= 1 - self.lmbda)[:, 0]
        logger.info(f'[self-training-strategy] Using {indices_selected.shape[0]} pseudo labels for self-training.')
        indices_self_train = indices_subsampled[indices_selected]

        return indices_self_train
