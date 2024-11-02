from dependency_injector import containers
from dependency_injector import providers

from active_learning_lab.config.active_learning.self_training._shared import (
    NUM_INITIAL_INSTANCES,
    NUM_QUERIES,
    RUNS,
    SEED,
    QUERY_SIZE,
    MINI_BATCH_SIZE,
    get_tmp_base,
    update_config as update_config_shared
)


EXPERIMENT_NAME = 'arr24-self-training'


DEFAULT_CONFIG = {
    'general': {
        'runs': RUNS,
        'seed': SEED,
        'max_reproducibility': True,
    },
    'active_learner': {
        # "default" for baselines, --active_learner self-training switches to self-training
        'active_learner_type': 'default',
        'active_learner_kwargs': {
            # self_training_method must be set
            # 'self_training_method': 'ust',
            'search_initial_model': False,
            'reuse_model': False,
        },
        'num_queries': NUM_QUERIES,
        'query_size': QUERY_SIZE,
        'query_strategy': 'random',
        'query_strategy_kwargs': {},
        'initialization_strategy': 'balanced',
        'initialization_strategy_kwargs': dict({
            'num_instances': NUM_INITIAL_INSTANCES
        }),
        'initial_model_selection': 0,
        'validation_set_sampling': 'stratified'
    },
    'classifier': {
        'classifier_name': 'setfit',
        'validation_set_size': 0.1,
        'classifier_kwargs': dict({
            'transformer_model': 'sentence-transformers/paraphrase-mpnet-base-v2',
            'mini_batch_size': MINI_BATCH_SIZE,
        })
    },
    'dataset': {
        'dataset_kwargs': {}
    }
}


TMP_BASE = get_tmp_base()


SETFIT_MAX_ITER = 1000


def set_defaults(config: providers.Configuration, override_args: dict) -> None:
    new_dict = DEFAULT_CONFIG

    if override_args['classifier']['classifier_name'] == 'setfit-ext':
        new_dict['classifier']['classifier_kwargs']['transformer_model'] = 'sentence-transformers/paraphrase-mpnet-base-v2'
        new_dict['classifier']['classifier_kwargs']['trainer_kwargs'] = {
            'num_epochs': 1,
            'learning_rate': 2e-5,
            'num_iterations': 1
        }

        new_dict['classifier']['classifier_kwargs']['model_kwargs'] = {
            'head_params': {
                'max_iter': SETFIT_MAX_ITER
            }
        }

    elif override_args['classifier']['classifier_name'] == 'transformer':
        new_dict['classifier']['classifier_kwargs']['transformer_model'] = 'bert-base-uncased'
        new_dict['classifier']['classifier_kwargs']['num_epochs'] = 15

    config.from_dict(new_dict)


def update_config(config: providers.Configuration) -> None:
    update_config_shared(config)


class Container(containers.DeclarativeContainer):
    pass
