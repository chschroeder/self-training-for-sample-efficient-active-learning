import logging
import numpy as np
from scipy.sparse import csr_matrix

from active_learning_lab.utils.numpy import get_class_histogram


def log_query(query_indices, dataset_config, max_results=5):
    text, y = dataset_config.train_raw.x, dataset_config.train_raw.y

    output = f'## Query Result [query_size={query_indices.shape[0]}]:\n'
    for i in query_indices[:max_results]:
        output += str(y[i]) + ' ' + str(text[i]) + '\n'

    logging.info(output)


def log_class_distribution(y, num_classes):
    if isinstance(y, csr_matrix):
        logging.info('Class Distribution (flattened):')
        logging.info(get_class_histogram(np.hstack(y.indices), num_classes, normalize=False))
    else:
        logging.info(f'Class Distribution: {get_class_histogram(y, num_classes, normalize=False)}')


def log_run_info(run_id, run_max, len_train, len_test):
    logging.info('#--------------------------------')
    logging.info('## Split: %d of %d', run_id, run_max)
    logging.info('##   Train %d / Test %d', len_train, len_test)
    logging.info('#--------------------------------')
