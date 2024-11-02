import copy
import glob
import logging

import torch

import numpy as np

from functools import partial
from pathlib import Path

from dependency_injector.wiring import inject, Provide
from torch.utils.tensorboard import SummaryWriter

from small_text import OverallUncertainty
from small_text.data.sampling import _get_class_histogram

from active_learning_lab.experiments.active_learning.active_learning_logging import log_run_info, log_query, \
    log_class_distribution
from active_learning_lab.experiments.active_learning.active_learning_tracking import ExperimentTracker, \
    RunResults
from active_learning_lab.experiments.active_learning.active_learning_experiment_helpers import get_active_learner
from active_learning_lab.experiments.active_learning.initialization import get_initial_indices
from active_learning_lab.experiments.active_learning.metrics import build_auc_metrics
from active_learning_lab.experiments.active_learning.results import ActiveLearningExperimentArtifacts
from active_learning_lab.experiments.active_learning.stopping_criteria import get_stopping_criteria_from_str

from active_learning_lab.utils.experiment import set_random_seed
from active_learning_lab.utils.labeling import label
from active_learning_lab.utils.pytorch import free_resources_fix
from active_learning_lab.utils.time import measure_time


INITIALIZATION_NUM_INSTANCES_DEFAULT = 25


class ExperimentConfig(object):
    """
    Experiment configuration.
    """
    def __init__(self, runs: int):
        """
        Parameters
        ----------
        runs : int
            Number of active learning runs (experiment repetitions).
        """
        self.runs = runs


class ClassificationConfig(object):
    """
    Classification configuration.
    """
    def __init__(self,
                 classifier_name: str,
                 classifier_factory,
                 classifier_kwargs: dict = None,
                 self_training: bool = False,
                 validation_set_size: float = 0.1):
        """
        Parameters
        ----------
        classifier_name : str
            Name of the classifier which is used by the classifier factory, the experiment
            config, and finally as part of the respective config's unique name in the results.
        classifier_factory : obj
            A classifier factory.
        classifier_kwargs : dict

        self_training : bool

        validation_set_size : float

        """
        self.classifier_name = classifier_name

        if classifier_kwargs is None:
            self.classifier_kwargs = dict()
        else:
            self.classifier_kwargs = classifier_kwargs

        self.classifier_factory = classifier_factory
        self.self_training = self_training
        self.validation_set_size = validation_set_size


class ActiveLearningConfig(object):
    """
    Active learning configuration.
    """
    def __init__(self,
                 active_learner_type: str,
                 active_learner_kwargs: dict,
                 num_queries: int,
                 query_strategy,
                 query_strategy_kwargs: dict,
                 query_size: int,
                 initialization_strategy,
                 initialization_strategy_kwargs,
                 stopping_criteria,
                 shared_initialization: bool = True,
                 validation_set_sampling: str = 'balanced'):
        """
        Parameters
        ----------
        active_learner_type : {'default', 'self-training'}
            Defines the active learning type. Usually you want 'default' here.
        active_learner_kwargs : dict

        num_queries : int
            Number of active learning queries.
        query_strategy : str
            String identifier for the query strategy to use
        query_strategy_kwargs : dict
            Kwargs for the query strategy.
        query_size : int
            Number of samples which are returned from one query step.
        initialization_strategy :

        initialization_strategy_kwargs :

        stopping_criteria : list of str or None

        reuse_model_across_queries : bool

        shared_initialization : bool

        validation_set_sampling : str

        """
        self.active_learner_type = active_learner_type
        self.active_learner_kwargs = active_learner_kwargs
        self.num_queries = num_queries
        self.query_strategy = query_strategy
        self.query_strategy_kwargs = query_strategy_kwargs
        self.query_size = query_size

        self.initialization_strategy = initialization_strategy
        self.initialization_strategy_kwargs = initialization_strategy_kwargs

        self.stopping_criteria = stopping_criteria

        self.reuse_model_across_queries = active_learner_kwargs.get('reuse_model', False)

        self.shared_initialization = shared_initialization
        self.validation_set_sampling = validation_set_sampling


class DatasetConfig(object):
    """
    Dataset configuration.
    """
    def __init__(self, dataset_name, dataset_kwargs, num_classes, train_raw=None, test_raw=None):
        self.dataset_name = dataset_name
        self.dataset_kwargs = dataset_kwargs
        self.num_classes = num_classes
        self.train_raw = train_raw
        self.test_raw = test_raw


class ActiveLearningRunConfig(object):

    def __init__(self, run_id, seed, exp_config, al_config, classification_config, dataset_config):
        self.run_id = run_id
        self.seed = seed
        self.exp_config = exp_config
        self.al_config = al_config
        self.classification_config = classification_config
        self.dataset_config = dataset_config


class EvaluationResult(object):

    def __init__(self,
                 y_test_pred,
                 y_test_proba,
                 y_train_labeled_pred,
                 y_train_labeled_proba,
                 y_train_pred,
                 y_train_proba,
                 y_valid_pred,
                 y_valid_proba):
        self.y_test_pred = y_test_pred
        self.y_test_proba = y_test_proba
        self.y_train_labeled_pred = y_train_labeled_pred
        self.y_train_labeled_proba = y_train_labeled_proba
        self.y_train_pred = y_train_pred
        self.y_train_proba = y_train_proba
        self.y_valid_pred = y_valid_pred
        self.y_valid_proba = y_valid_proba


class ActiveLearningExperiment(object):

    def __init__(self, exp_config, al_config, classification_config, dataset_config, train, tmp_dir,
                 gpu=None):
        """
        Parameters
        ----------
        exp_config : ExperimentConfig

        al_config : ActiveLearningConfig

        classification_cfg : ClassificationConfig

        dataset_config : DatasetConfig

        n_queries : int
            number of queries
        query_size : int
            number of queried examples
        """

        self.exp_config = exp_config
        self.al_config = al_config
        self.classification_cfg = classification_config
        self.dataset_config = dataset_config

        self.num_classes = self.dataset_config.num_classes

        self.tmp_dir = Path(tmp_dir)

        logging.info('Initialization [strategy=%s, shared=%s]', str(self.al_config.initialization_strategy),
                     str(self.al_config.shared_initialization))
        logging.info('Query strategy: %s', str(self.al_config.query_strategy))
        logging.info('Train dataset [length={}, num_classes={}, type={}]'
                     .format(len(train), self.num_classes, str(type(train))))

        self.gpu = gpu

        self.experiment_tracker = ExperimentTracker(self.num_classes)

    @inject
    def run(self,
            train_set,
            valid_set,
            test_set,
            summary_writer: SummaryWriter = Provide['summary_writer']):
        """
        Runs the active learning experiment. An experiment consists of several repetitions.

        Parameters
        ----------
        train_set :

        test_set :


        Returns
        -------
        result : ActiveLearningExperimentArtifacts

        """
        artifacts = self._pre_experiment(train_set, test_set)

        # fix the initial indices so that all runs have the same starting conditions
        self.indices_initial = self._get_initial_indices(train_set)

        # fix the seeds here, so that different runs start with exactly the same situation
        seeds = [np.random.randint(2*+32) for _ in range(self.exp_config.runs)]

        self.experiment_tracker.summary_writer = summary_writer
        for run_id in range(1, self.exp_config.runs+1):
            assert run_id == self.experiment_tracker.run_id

            run_config = ActiveLearningRunConfig(run_id,
                                                 seeds[run_id-1],
                                                 self.exp_config,
                                                 self.al_config,
                                                 self.classification_cfg,
                                                 self.dataset_config)
            self._pre_run(run_id, train_set, test_set)
            artifacts += ActiveLearningRun(
                run_config,
                self.experiment_tracker,
                self.num_classes,
                self.tmp_dir,
                self.indices_initial
            ).execute(train_set, valid_set, test_set)
            self._post_run()

        self.results = self._post_experiment(artifacts)
        return self.results

    def _get_initial_indices(self, train_set):

        num_samples = self.al_config.initialization_strategy_kwargs.get('num_instances',
                                                                        INITIALIZATION_NUM_INSTANCES_DEFAULT)

        if '.' in str(num_samples):
            num_samples = float(num_samples)
            if num_samples > 1.0:
                raise ValueError('Relative initial pool size (num_samples) too large!')
            num_samples = int(len(train_set) * num_samples)
        else:
            num_samples = int(num_samples)
            assert num_samples > 0

        if self.al_config.shared_initialization is True:
            indices_initial = get_initial_indices(train_set,
                                                  self.dataset_config.train_raw,
                                                  self.al_config.initialization_strategy,
                                                  self.al_config.initialization_strategy_kwargs,
                                                  num_samples)
        else:
            indices_initial = None

        return indices_initial

    def _pre_experiment(self, train_set, test_set):

        np.savez(self.tmp_dir.joinpath('train_labels.npz'), vectors=train_set.y)
        np.savez(self.tmp_dir.joinpath('test_labels.npz'), vectors=test_set.y)

        artifacts = [('train_labels.npz', self.tmp_dir.joinpath('train_labels.npz')),
                     ('test_labels.npz', self.tmp_dir.joinpath('test_labels.npz'))]

        return artifacts

    def _pre_run(self, run_id, train_set, test_set):
        log_run_info(run_id, self.exp_config.runs, len(train_set), len(test_set))

    def _post_run(self):
        self.experiment_tracker.next_run()
        if torch.cuda.is_available():
            free_resources_fix(full=True)

    def _post_experiment(self, artifacts):

        results_file = self.experiment_tracker.metrics_tracker.write(self.tmp_dir.joinpath('results.csv').resolve())
        results_agg_file = self.experiment_tracker.metrics_tracker.write_aggregate(
            self.tmp_dir.joinpath('results_agg.csv').resolve())

        artifacts += [
            ('results.csv', results_file),
            ('results_agg.csv', results_agg_file)
        ]

        if self.experiment_tracker.stopping_criteria_results is not None:
            stopping_criteria_results_file = self.experiment_tracker.write_stopping_criteria_results(
                self.tmp_dir.joinpath('stopping_criteria.csv').resolve())
            artifacts += [
                ('stopping_criteria.csv', stopping_criteria_results_file)
            ]


        if self.al_config.num_queries > 1:
            artifacts = build_auc_metrics(self.tmp_dir, self.experiment_tracker.metrics_tracker, artifacts)
        else:
            artifacts = []

        artifacts += [
            ('random_states.npz', self.experiment_tracker.track_random_states(self.tmp_dir))
        ]

        return ActiveLearningExperimentArtifacts(artifacts)


class ActiveLearningRun(object):

    def __init__(self,
                 run_config: ActiveLearningRunConfig,
                 experiment_tracker: ExperimentTracker,
                 num_classes: int,
                 tmp_dir,
                 indices_initial):

        self.run_config = run_config
        self.experiment_tracker = experiment_tracker

        self.num_classes = num_classes
        self.tmp_dir = tmp_dir

        self.indices_initial = indices_initial

        if self.run_config.al_config.stopping_criteria is not None:
            stopping_criteria = self.run_config.al_config.stopping_criteria
            self.stopping_criteria = get_stopping_criteria_from_str(stopping_criteria, num_classes)
        else:
            self.stopping_criteria = None
        self.embeddings = None

        self._prepare_folders()

    def _prepare_folders(self):
        self.run_subdir = self.tmp_dir.joinpath('run-' + str(self.run_config.run_id))
        self.run_subdir.mkdir(parents=True)

        self.train_history_dir = self.run_subdir.joinpath('train-history')
        self.train_history_dir.mkdir()

        if self.run_config.al_config.active_learner_type == 'self-training':
            self.self_training_dir = self.run_subdir.joinpath('self-training')
            self.self_training_dir.mkdir()

    def execute(self, train_set, valid_set, test_set):
        """
        Executes a single active learning run.
        """
        set_random_seed(self.run_config.seed)

        indices_initial = self._get_initial_indices(train_set)
        active_learner = get_active_learner(self.run_config,
                                            self.num_classes,
                                            train_set,
                                            test_set)

        # Initial evaluation
        al_kwargs = self.run_config.al_config.active_learner_kwargs
        label_noise = float(al_kwargs['label_noise']) if 'label_noise' in al_kwargs else None
        self.run_initial_evaluation(active_learner,
                                    train_set,
                                    indices_initial,
                                    valid_set,
                                    test_set,
                                    label_noise=label_noise)

        #
        # main loop
        #
        for q in range(1, self.run_config.al_config.num_queries+1):
            ind, scores, run_results = self.run_query(active_learner,
                                                      q,
                                                      train_set,
                                                      valid_set,
                                                      test_set,
                                                      label_noise=label_noise)
            logging.info(f'Query {q} class distribution: '
                         f'{_get_class_histogram(train_set.y[ind], active_learner.classifier.num_classes)}')

            if self.stopping_criteria is not None:
                self.evaluate_stopping_criteria(active_learner, q, run_results)

            self.post_query(active_learner, q)

        return self._create_artifacts(active_learner)

    def _get_initial_indices(self, train_set):
        num_samples = self.run_config.al_config.initialization_strategy_kwargs.get('num_instances',
                                                                                   INITIALIZATION_NUM_INSTANCES_DEFAULT)
        if self.indices_initial is None or self.run_config.al_config.shared_initialization is False:
            self.indices_initial = get_initial_indices(train_set,
                                                       self.run_config.al_config.initialization_strategy,
                                                       num_samples)

            logging.info('Selecting initial indices...')

        logging.info(f'Initial indices ({self.indices_initial.shape[0]}: {self.indices_initial}')
        return self.indices_initial

    def post_query(self, active_learner, q):

        # track metrics
        if hasattr(active_learner.classifier, 'train_history'):
            if active_learner.classifier.train_history is not None:
                self.experiment_tracker.metrics_tracker.track_train_history(
                    q, self.train_history_dir, active_learner.classifier.train_history)

        free_resources_fix(full=True)

    def _create_artifacts(self, active_learner):

        queried_indices_file_path = self.run_subdir.joinpath('queries.npz')
        self.experiment_tracker.query_tracker.write(queried_indices_file_path)

        artifacts = [('queries.npz', queried_indices_file_path)]

        if self.run_config.al_config.active_learner_type == 'self-training':
            self_training_overall_path = self.run_subdir.joinpath('self-training/self-training-overall.csv')
            self_training_detailed_path = self.run_subdir.joinpath('self-training/self-training-detailed.csv')
            weights_path = self.run_subdir.joinpath('self-training/weights.npz')

            active_learner.self_training_tracking.write(self_training_overall_path,
                                                        self_training_detailed_path,
                                                        weights_path)

        for f in glob.glob(str(self.run_subdir) + '/**/*', recursive=True):
            artifacts.append((f, self.run_subdir.joinpath(f)))

        return artifacts

    def run_initial_evaluation(self, active_learner, train_set, indices_initial, valid_set,
                               test_set, label_noise=None):

        y_train_true = train_set.y
        y_train_labeled_true = train_set[indices_initial].y

        # rename to warmstart?
        if not 'disable_initial_training' in self.run_config.classification_config.classifier_kwargs or \
                self.run_config.classification_config.classifier_kwargs['disable_initial_training'] == False:

            y_init = label(y_train_true[self.indices_initial],
                           self.num_classes,
                           label_noise=label_noise)
            y_tmp = active_learner.dataset.y
            y_tmp_initial = np.copy(y_tmp[indices_initial])
            y_tmp[indices_initial] = y_init
            active_learner.dataset.y = y_tmp

            active_learner.initialize(clf_or_indices=self.indices_initial, retrain=False)

            y_tmp = active_learner.dataset.y
            y_tmp[indices_initial] = y_tmp_initial
            active_learner.dataset.y = y_tmp

            indices_validation = None

            logging.info('Initital training starting...')
            update_time = measure_time(
                partial(active_learner._retrain, indices_validation=indices_validation),
                has_return_value=False)
            logging.info('Initital training finished...')
        else:
            y_init = np.array([], dtype=int)
            self.indices_initial = np.array([], dtype=int)
            active_learner.initialize_data(np.array([], dtype=int), y_init, retrain=False)

            update_time = 0
            active_learner._clf = active_learner._clf_factory.new()

        eval_result = self._evaluate(active_learner.classifier, train_set, valid_set, test_set,
                                     self.indices_initial)

        self.experiment_tracker.track_predict_proba(0, eval_result.y_test_proba)
        np.savez(str(self.run_subdir.joinpath(f'predictions_0.npz')),  # run_id starts at 1,
                 train_predictions=eval_result.y_train_pred,  # so 0 is the initial value
                 train_proba=eval_result.y_train_proba,
                 test_predictions=eval_result.y_test_pred,
                 test_proba=eval_result.y_test_proba, )

        run_results = RunResults(0,
                                 update_time,
                                 y_train_true,
                                 eval_result.y_train_pred,
                                 eval_result.y_train_proba,
                                 y_train_labeled_true,
                                 eval_result.y_train_labeled_pred,
                                 eval_result.y_train_labeled_proba,
                                 valid_set.y,
                                 eval_result.y_valid_pred,
                                 eval_result.y_valid_proba,
                                 test_set.y,
                                 eval_result.y_test_pred,
                                 eval_result.y_test_proba)

        self.experiment_tracker.track_initialization_results(indices_initial,
                                                             y_init,
                                                             run_results)

    def run_query(self, active_learner, q, train_set, valid_set, test_set, label_noise=None):

        logging.info(f'## Run: %d / Query: %d '
                     f'[labeled_pool_size={active_learner.indices_labeled.shape[0]}]',
                     self.run_config.run_id, q)

        indices_labeled = active_learner.indices_labeled

        if indices_labeled.shape[0] == 0:
            train_labeled, y_train_labeled_true = np.array([], dtype=int), np.array([], dtype=int)
        else:
            train_labeled, y_train_labeled_true = train_set[indices_labeled], train_set[indices_labeled].y

        query_kwargs = copy.deepcopy(self.run_config.al_config.query_strategy_kwargs)
        for filtered_key in ['subsample']:
            if filtered_key in query_kwargs:
                del query_kwargs[filtered_key]

        query_func = partial(active_learner.query, num_samples=self.run_config.al_config.query_size,
                             query_strategy_kwargs=query_kwargs)
        query_time, ind = measure_time(query_func)

        log_query(ind, self.run_config.dataset_config)
        log_class_distribution(active_learner.y, self.num_classes)

        query_strategy = active_learner.query_strategy

        # Update
        if self.run_config.classification_config.classifier_name in ['transformer', 'kimcnn']:
            active_learner.classifier.model.zero_grad(set_to_none=True)
            if not active_learner.reuse_model:
                active_learner.classifier.model = None
        elif self.run_config.classification_config.classifier_name.startswith('setfit'):
            active_learner.classifier.model.model_body.zero_grad(set_to_none=True)
            if not active_learner.reuse_model:
                active_learner.classifier.model = None
        free_resources_fix(full=True)

        indices_labeled_and_update = np.concatenate((active_learner.indices_labeled, ind))

        run_results, scores = self._update_and_evaluate(active_learner,
                                                        ind,
                                                        q,
                                                        query_strategy,
                                                        query_time,
                                                        test_set,
                                                        train_set,
                                                        valid_set,
                                                        indices_labeled_and_update,
                                                        indices_labeled,
                                                        y_train_labeled_true,
                                                        train_set.y,
                                                        label_noise=label_noise)

        self.experiment_tracker.track_query(ind,
                                            train_set.y[ind],
                                            scores,
                                            run_results,
                                            q,
                                            active_learner.indices_labeled.shape[0])

        return ind, scores, run_results

    def _update_and_evaluate(self, active_learner, ind, q, query_strategy, query_time, test_set,
                             train_set, valid_set, indices_labeled_and_update,
                             indices_labeled, y_train_labeled_true, y_train_true, label_noise=None):

        y_train_update = label(y_train_true[ind], self.num_classes, label_noise=label_noise)

        indices_validation = None
        update_func = partial(active_learner.update, y_train_update, indices_validation=indices_validation)
        update_time = measure_time(update_func, has_return_value=False)
        # print(active_learner.y, self.num_classes)

        log_class_distribution(active_learner.y, self.num_classes)
        self.experiment_tracker.track_labeled_pool(y_train_true[indices_labeled_and_update], self.num_classes, q)

        # Evaluation
        eval_result = self._evaluate(active_learner.classifier, train_set, valid_set, test_set, indices_labeled)
        self.experiment_tracker.track_predicted_labels(q,
                                                       self.num_classes,
                                                       eval_result.y_train_pred,
                                                       eval_result.y_test_pred,
                                                       eval_result.y_train_labeled_pred)

        self.experiment_tracker.track_predict_proba(q, eval_result.y_test_proba)
        np.savez(str(self.run_subdir.joinpath(f'predictions_{q}.npz')),
                 train_predictions=eval_result.y_train_pred,
                 train_proba=eval_result.y_train_proba,
                 test_predictions=eval_result.y_test_pred,
                 test_proba=eval_result.y_test_proba,
                 valid_predictions=eval_result.y_valid_pred,
                 valid_proba=eval_result.y_valid_proba)

        scores = query_strategy.scores_ if hasattr(query_strategy, 'scores_') else None
        if scores is not None:
            self.experiment_tracker.track_query_scores(
                q,
                active_learner.query_strategy,
                self.num_classes,
                scores,
                eval_result.y_train_pred
            )

        run_results = RunResults(query_time,
                                 update_time,
                                 y_train_true,
                                 eval_result.y_train_pred,
                                 eval_result.y_train_proba,
                                 y_train_labeled_true,
                                 eval_result.y_train_labeled_pred,
                                 eval_result.y_train_labeled_proba,
                                 valid_set.y,
                                 eval_result.y_valid_pred,
                                 eval_result.y_valid_proba,
                                 test_set.y,
                                 eval_result.y_test_pred,
                                 eval_result.y_test_proba)

        return run_results, scores

    def _evaluate(self, clf, train_set, valid_set, test_set, indices_labeled) -> EvaluationResult:
        y_train_pred, y_train_proba = clf.predict(train_set, return_proba=True)
        y_train_labeled_pred, y_train_labeled_proba = y_train_pred[indices_labeled], \
                                                      y_train_proba[indices_labeled]

        y_valid_pred, y_valid_proba = clf.predict(valid_set, return_proba=True)
        y_test_pred, y_test_proba = clf.predict(test_set, return_proba=True)

        return EvaluationResult(y_test_pred,
                                y_test_proba,
                                y_train_labeled_pred,
                                y_train_labeled_proba,
                                y_train_pred,
                                y_train_proba,
                                y_valid_pred,
                                y_valid_proba)

    def evaluate_stopping_criteria(self, active_learner, query_id, run_results):

        stop_results = []
        for name, criterion in self.stopping_criteria.items():

            if isinstance(criterion, OverallUncertainty):
                indices_stopping = np.arange(run_results.y_train_pred.shape[0])
                stop_result = criterion.stop(active_learner=active_learner,
                                             predictions=run_results.y_train_pred,
                                             proba=run_results.y_train_pred_proba,
                                             indices_stopping=indices_stopping)
            else:
                stop_result = criterion.stop(active_learner=active_learner,
                                             predictions=run_results.y_train_pred,
                                             proba=run_results.y_train_pred_proba)
            logging.info(f'Stopping Criterion ({name}): result={stop_result}')

            score = np.nan if not hasattr(criterion, 'score_') else criterion.score_
            stop_results.append([name, stop_result, score])

        self.experiment_tracker.track_stopping_criteria(query_id, stop_results)
