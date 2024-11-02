import gc
import logging
import torch

import numpy as np

from pathlib import Path

from dependency_injector.wiring import inject

from small_text import SetFitClassification

from small_text.active_learner import PoolBasedActiveLearner
from small_text.data.sampling import _get_class_histogram
from small_text.training.early_stopping import EarlyStopping
from small_text.training.metrics import Metric

from active_learning_lab.experiments.active_learning.self_training.utils.setfit import apply_setfit_memory_fix


logger = logging.getLogger(__name__)


class PoolBasedActiveLearnerWithSelfTraining(PoolBasedActiveLearner):

    @inject
    def __init__(self, clf_factory, query_strategy, dataset, self_trainer, tmp_dir,
                 reuse_model=False, self_training_tracking=None):

        super().__init__(clf_factory, query_strategy, dataset,
                         reuse_model=reuse_model)

        self.self_trainer = self_trainer
        self.self_training_tracking = self_training_tracking

        self.self_trainer.self_training_tracking = self_training_tracking

        self.query_id = 0

        # tmp_dir_: tmp dir as string without trailing slash
        self.tmp_dir_ = tmp_dir
        self.tmp_dir = self.tmp_dir_.name
        Path(self.tmp_dir).mkdir(exist_ok=True)

        self._set_setfit_trainer_defaults()

        logger.info(f'[self-training] {str(self)}')

    def _set_setfit_trainer_defaults(self):
        from setfit.trainer import SetFitTrainer
        from active_learning_lab.thirdparty.setfit.trainer import SetFitTrainerExtended

        defaults = list(SetFitTrainer.__init__.__defaults__)
        # 7 is the index for num_iterations
        defaults[7] = 5  # set num_iterations=5
        SetFitTrainer.__init__.__defaults__ = tuple(defaults)

        defaults = list(SetFitTrainerExtended.__init__.__defaults__)
        defaults[7] = 5
        SetFitTrainerExtended.__init__.__defaults__ = tuple(defaults)

        logger.info(f'__defaults__: {SetFitTrainer.__init__.__defaults__}')

    def query(self, num_samples=10, representation=None, query_strategy_kwargs=dict()):

        result = super().query(num_samples=num_samples, representation=representation,
                               query_strategy_kwargs=query_strategy_kwargs)
        self.query_id += 1
        return result

    def update(self, y, indices_validation=None):
        super().update(y, indices_validation=indices_validation)

    def _retrain(self, indices_validation=None):
        logger.info(f'[self-training] use_validation_set: {indices_validation is not None}')
        if indices_validation is None:
            train_set = self.dataset[self.indices_labeled].clone()
            train_set.y = self.y
            validation_set = None
        else:
            mask = np.ones((self.indices_labeled.shape[0],)).astype(bool)
            mask[indices_validation] = False

            train_set = self.dataset[self.indices_labeled[mask]].clone()
            train_set.y = self.y
            validation_set = train_set[indices_validation].clone()

        # sub_train = self.dataset[self.indices_labeled]
        # logger.info(f'[self-training] _retrain labels: {_get_class_histogram(sub_train.y, self._clf.num_classes)}')
        # logger.info(f'[self-training] _retrain labels (validation only): {_get_class_histogram(sub_train[indices_validation].y, self._clf.num_classes)}')

        if self._clf is None or not self.reuse_model:
            if hasattr(self, '_clf'):
                del self._clf
            self._clf = self._clf_factory.new()

            if hasattr(self._clf, 'validations_per_epoch'):
                early_stopping = EarlyStopping(Metric('val_loss', lower_is_better=True),
                                               patience=5 * self._clf.validations_per_epoch)

                if isinstance(self._clf, SetFitClassification):
                    self._clf = apply_setfit_memory_fix(self._clf, mini_batch_size=self._clf.mini_batch_size)

                self._clf.fit(train_set, validation_set, early_stopping=early_stopping)
            else:
                if isinstance(self._clf, SetFitClassification):
                    self._clf = apply_setfit_memory_fix(self._clf, mini_batch_size=self._clf.mini_batch_size)

                self._clf.fit(train_set, validation_set)

        gc.collect()
        torch.cuda.empty_cache()

        if self._clf.model is not None:
            size = len(self.dataset)
            mask = np.ones(size, bool)
            mask[np.concatenate([self.indices_labeled, self.indices_ignored])] = False
            indices_unlabeled = np.arange(size)[mask]

            results = self.self_trainer.train(self._clf,
                                              self.dataset,
                                              self.y,
                                              indices_unlabeled,
                                              self.indices_labeled,
                                              indices_validation,
                                              test_set=self.test_set)

            self.self_training_tracking.track(self.query_id, results)

            gc.collect()
            torch.cuda.empty_cache()

    def __str__(self):
        return f'PoolBasedActiveLearnerWithSelfTraining(self_trainer={self.self_trainer})'

    def __del__(self):
        self.tmp_dir_.cleanup()
