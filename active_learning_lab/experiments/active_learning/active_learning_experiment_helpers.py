import mlflow
import tempfile
from small_text.active_learner import PoolBasedActiveLearner

from active_learning_lab.experiments.active_learning.self_training.active_learner import (
    PoolBasedActiveLearnerWithSelfTraining
)
from active_learning_lab.experiments.active_learning.self_training.strategies.actune import AcTune
from active_learning_lab.experiments.active_learning.self_training.strategies.ust import UST
from active_learning_lab.experiments.active_learning.self_training.strategies.nest import NEST
from active_learning_lab.experiments.active_learning.self_training.strategies.verips import VERIPS
from active_learning_lab.experiments.active_learning.self_training.strategies.hast import HAST

from active_learning_lab.experiments.active_learning.strategies import query_strategy_from_str


def get_active_learner(run_config, num_classes, train_set, test_set):
    if isinstance(run_config.al_config.query_strategy, str):
        query_strategy = query_strategy_from_str(run_config.al_config.query_strategy,
                                                 {},
                                                 num_classes)
    else:
        raise NotImplementedError('Todo: Load query strategy from context')

    active_learner_type = run_config.al_config.active_learner_type
    if active_learner_type == 'default':
        active_learner = PoolBasedActiveLearner(
            run_config.classification_config.classifier_factory,
            query_strategy,
            train_set,
            reuse_model=run_config.al_config.reuse_model_across_queries)
    elif active_learner_type == 'self-training':
        from active_learning_lab.experiments.active_learning.self_training.tracking import SelfTrainingExperimentTracker
        self_training_tracking = SelfTrainingExperimentTracker(run_config.run_id, num_classes)

        if 'self_training_method' not in run_config.al_config.active_learner_kwargs:
            raise ValueError('No self-training method specified')

        al_kwargs = run_config.al_config.active_learner_kwargs
        tmp_dir = tempfile.TemporaryDirectory()

        mlflow.log_param('self_training_method', al_kwargs['self_training_method'])

        if al_kwargs['self_training_method'] == 'verips-e':
            self_trainer = VERIPS('entropy', self_training_iterations=1, subsample_size=16384)
        elif al_kwargs['self_training_method'] == 'verips-m':
            self_trainer = VERIPS('margin', self_training_iterations=1, subsample_size=16384)
        elif al_kwargs['self_training_method'] == 'actune':
            self_trainer = AcTune(self_training_iterations=1)
        elif al_kwargs['self_training_method'] == 'ust':
            self_trainer = UST(self_training_iterations=1, subsample_size=16384)
        elif al_kwargs['self_training_method'] == 'nest':
            self_trainer = NEST(self_training_iterations=1, subsample_size=16384)
        elif al_kwargs['self_training_method'] == 'hast':
            self_trainer = HAST(self_training_iterations=1, subsample_size=16384, use_class_weights=True, labeled_to_unlabeled_factor=0.1)
        elif al_kwargs['self_training_method'] == 'hast-without-class-weights':
            self_trainer = HAST(self_training_iterations=1, subsample_size=16384, use_class_weights=False, labeled_to_unlabeled_factor=0.1)
        elif al_kwargs['self_training_method'] == 'hast-without-pseudo-label-penalty':
            self_trainer = HAST(self_training_iterations=1, subsample_size=16384, use_class_weights=True, labeled_to_unlabeled_factor=1)
        elif al_kwargs['self_training_method'] == 'hast-without-class-weights-and-pseudo-label-penalty':
            self_trainer = HAST(self_training_iterations=1, subsample_size=16384, use_class_weights=False, labeled_to_unlabeled_factor=1)
        else:
            raise ValueError(f'Invalid self-training method specified: '
                             f'{al_kwargs["self_training_method"]}')

        active_learner = PoolBasedActiveLearnerWithSelfTraining(
            run_config.classification_config.classifier_factory,
            query_strategy,
            train_set,
            self_trainer,
            tmp_dir,
            reuse_model=run_config.al_config.reuse_model_across_queries,
            self_training_tracking=self_training_tracking
        )

        # test set is only used for intermediate evaluations of self-training where we check
        #  if self-training has a benefit over the initial model
        # TODO: work out a better way to pass this
        active_learner.test_set = test_set
    else:
        raise NotImplementedError(f'Invalid active_learner_type: {active_learner_type}')

    return active_learner
