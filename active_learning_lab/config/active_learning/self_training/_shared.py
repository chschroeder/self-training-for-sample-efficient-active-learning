import mlflow
import socket

from dependency_injector import providers


EXPERIMENT_VERSION = '1.0.0'

#
# General
#
RUNS = 5

# obtained via np.random.randint(2**32)
SEED = 762671998

#
# Active Learning
#

NUM_INITIAL_INSTANCES = 30

NUM_QUERIES = 10

QUERY_SIZE = 10

#
# SetFit / Vanilla Transformer
#

MINI_BATCH_SIZE = 24


def get_tmp_base():
    return "/tmp/active-learning-lab-v2"


def update_config(config: providers.Configuration) -> None:

    mlflow.log_param('experiment_version', EXPERIMENT_VERSION)

    if config['classifier']['classifier_name'] in ['transformer', 'setfit-ext']:
        if config['dataset']['dataset_name'] == 'mr':
            config['dataset']['dataset_kwargs']['max_length'] = 64
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 40
        elif config['dataset']['dataset_name'] == 'trec':
            config['dataset']['dataset_kwargs']['max_length'] = 64
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 40
        elif config['dataset']['dataset_name'] == 'ag-news':
            config['dataset']['dataset_kwargs']['max_length'] = 64
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 40
        elif config['dataset']['dataset_name'] == 'imdb':
            config['dataset']['dataset_kwargs']['max_length'] = 512
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 14
        elif config['dataset']['dataset_name'] == 'dbp-140k':
            config['dataset']['dataset_kwargs']['max_length'] = 128
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 24
        elif config['dataset']['dataset_name'] == 'yah-140k':
            config['dataset']['dataset_kwargs']['max_length'] = 512
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 14
