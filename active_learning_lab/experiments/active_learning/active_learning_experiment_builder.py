import logging
from functools import partial

from active_learning_lab.classification.factories.factories_base import get_factory, FactoryWrappingApplier
from active_learning_lab.classification.fit_wrappers import wrap_clf
from active_learning_lab.data.dataset_loaders import DatasetLoader

from active_learning_lab.experiments.active_learning.active_learning_experiment import ActiveLearningExperiment, \
    ClassificationConfig, ActiveLearningConfig, ExperimentConfig, DatasetConfig
from active_learning_lab.utils.experiment import get_data_dir

from small_text import EarlyStopping, Metric
from small_text.integrations.pytorch.classifiers.base import AMPArguments


class ActiveLearningExperimentBuilder(object):

    def __init__(self, active_learner_type, active_learner_kwargs, num_queries, query_size,
                 query_strategy, query_strategy_kwargs, validation_set_sampling, runs, tmp_dir):
        self.active_learner_kwargs = active_learner_kwargs
        self.num_queries = num_queries
        self.query_size = query_size
        if '.' in str(self.query_size):
            self.query_size = float(self.query_size)
            if self.query_size * self.num_queries > 1.0:
                raise ValueError('Relative query size too large! Reduce query_size or num_queries.')
            assert 1 > self.query_size > 0
        else:
            self.query_size = int(self.query_size)
            assert self.query_size > 0

        self.query_strategy = query_strategy
        self.query_strategy_kwargs = query_strategy_kwargs
        self.validation_set_sampling = validation_set_sampling

        self.runs = runs
        self.active_learner_type = active_learner_type
        self.tmp_dir = tmp_dir

        self.classifier_name = None
        self.classifier_kwargs = None
        self.classifier_factory = None
        self.train = None
        self.test = None
        self.initialization_strategy = None
        self.initialization_strategy_kwargs = None

    def with_dataset(self, dataset_name, dataset_kwargs):

        if self.classifier_name is None:
            raise ValueError('classifier_name must be set prior to assigning the dataset_name')

        if dataset_kwargs is None:
            dataset_kwargs = dict()

        if self.classifier_name == 'transformer':
            if not 'transformer_model' in self.classifier_kwargs:
                raise RuntimeError('Key \'transformer_model\' not set in transformer_model. '
                                   'This should not happen.')

            dataset_kwargs['tokenizer_name'] = self.classifier_kwargs['transformer_model']

        loader = DatasetLoader(data_dir=get_data_dir())
        train_raw, valid_raw, test_raw, num_classes = loader.load_dataset(dataset_name,
                                                                          dataset_kwargs,
                                                                          self.classifier_name,
                                                                          self.classifier_kwargs,
                                                                          dataset_type='raw')

        self.train, self.valid, self.test, _ = loader.load_dataset(dataset_name,
                                                                   dataset_kwargs,
                                                                   self.classifier_name,
                                                                   self.classifier_kwargs)

        self.num_classes = num_classes
        logging.info(f'Dataset info [train_size={len(self.train)}, test_size={len(self.test)}]')

        if isinstance(self.query_size, float):
            base_size = len(self.train)
            self.query_size = int(base_size * self.query_size)
            logging.info(
                f'Resolve relative query size: {self.query_size} (base size: {base_size})')

        self.dataset_config = DatasetConfig(dataset_name, dataset_kwargs, num_classes, train_raw, test_raw)

        return self

    def with_classifier(self, classifier_name, validation_set_size, classifier_kwargs):

        self.classifier_name = classifier_name
        self.validation_set_size = validation_set_size

        self.classifier_kwargs = dict() if classifier_kwargs is None else classifier_kwargs

        self.incremental_training = self.classifier_kwargs.get('incremental_training', False)
        if 'incremental_training' in self.classifier_kwargs:
            del self.classifier_kwargs['incremental_training']

        if 'amp_args' in self.classifier_kwargs:
            self.classifier_kwargs['amp_args'] = AMPArguments(**self.classifier_kwargs['amp_args'])

        if self.classifier_name == 'transformer':
            if not 'transformer_model' in self.classifier_kwargs:
                raise ValueError('\'transformer_model\' not set in classifier_kwargs')

        return self

    def with_initialization(self, initialization_strategy, initialization_strategy_kwargs):
        self.initialization_strategy = initialization_strategy
        if initialization_strategy_kwargs is None:
            self.initialization_strategy_kwargs = dict()
        else:
            self.initialization_strategy_kwargs = initialization_strategy_kwargs
        return self

    def with_stopping_criteria(self, stopping_criteria):
        self.stopping_criteria = stopping_criteria
        return self

    def build(self):
        exp_args = ExperimentConfig(self.runs)

        if 'reuse_model_across_queries' in self.classifier_kwargs:
            raise ValueError('Deprecated parameter: reuse_model_across_queries')

        al_cfg = ActiveLearningConfig(self.active_learner_type,
                                      self.active_learner_kwargs,
                                      self.num_queries,
                                      self.query_strategy,
                                      self.query_strategy_kwargs,
                                      self.query_size,
                                      self.initialization_strategy,
                                      self.initialization_strategy_kwargs,
                                      self.stopping_criteria,
                                      validation_set_sampling=self.validation_set_sampling)

        if self.classifier_factory is None:

            self.classifier_factory = get_factory(self.classifier_name,
                                                  self.num_classes,
                                                  classifier_kwargs=self.classifier_kwargs)

            if 'early_stopping' in self.classifier_kwargs:
                if 'val_loss' in self.classifier_kwargs['early_stopping']:
                    patience = int(self.classifier_kwargs['early_stopping']['val_loss'])
                    early_stopping = EarlyStopping(Metric('val_loss'), patience=patience)
                    wrappers = [partial(wrap_clf, early_stopping=early_stopping)]
                else:
                    raise ValueError(f'Unknown early stopping config: '
                                     f'{self.classifier_kwargs["early_stopping"]}')

                del self.classifier_kwargs['early_stopping']

                self.classifier_factory = FactoryWrappingApplier(self.classifier_factory, wrappers)

        classification_config = ClassificationConfig(self.classifier_name,
                                                     self.classifier_factory,
                                                     classifier_kwargs=self.classifier_kwargs,
                                                     validation_set_size=self.validation_set_size)

        return ActiveLearningExperiment(exp_args,
                                        al_cfg,
                                        classification_config,
                                        self.dataset_config,
                                        self.train,
                                        self.tmp_dir)


def get_active_learner_builder(config, tmp_dir):

    if not 'stopping_criteria' in config['active_learner']:
        config['active_learner']['stopping_criteria'] = None

    builder = ActiveLearningExperimentBuilder(config['active_learner']['active_learner_type'],
                                              config['active_learner']['active_learner_kwargs'],
                                              config['active_learner']['num_queries'],
                                              config['active_learner']['query_size'],
                                              config['active_learner']['query_strategy'],
                                              config['active_learner']['query_strategy_kwargs'],
                                              config['active_learner']['validation_set_sampling'],
                                              config['general']['runs'],
                                              str(tmp_dir)) \
        .with_classifier(config['classifier']['classifier_name'],
                         config['classifier']['validation_set_size'],
                         config['classifier']['classifier_kwargs']) \
        .with_initialization(config['active_learner']['initialization_strategy'],
                             config['active_learner']['initialization_strategy_kwargs']) \
        .with_stopping_criteria(config['active_learner']['stopping_criteria']) \
        .with_dataset(config['dataset']['dataset_name'], config['dataset']['dataset_kwargs'])

    return builder
