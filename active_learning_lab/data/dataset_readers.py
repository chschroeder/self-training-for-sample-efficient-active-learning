import logging

import numpy as np
from pathlib import Path

from small_text.data import TextDataset
from small_text.utils.labels import list_to_csr

from active_learning_lab.data.dataset_abstractions import RawDataset
from active_learning_lab.data.dataset_preprocessing import (
    _get_huggingface_tokenizer,
    _text_to_bow,
    _text_to_huggingface,
    _text_to_tps)
from active_learning_lab.data.datasets import DataSets, DataSetType, DatasetReaderNotFoundException
from active_learning_lab.utils.data import get_validation_set


def _read_trec(dataset_name: str, dataset_kwargs: dict, _: str,
               classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    trec_dataset = datasets.load_dataset('trec')
    num_classes = 6

    label_col = 'coarse_label'

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        train, test = _text_to_huggingface(tokenizer,
                                           trec_dataset['train']['text'],
                                           trec_dataset['train'][label_col],
                                           trec_dataset['test']['text'],
                                           trec_dataset['test'][label_col],
                                           num_classes,
                                           int(dataset_kwargs['max_length']))

        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.RAW:
        train, test = RawDataset(np.array(trec_dataset['train']['text']),
                                 np.array(trec_dataset['train'][label_col]),
                                 target_labels=np.arange(num_classes)), \
                      RawDataset(np.array(trec_dataset['test']['text']),
                                 np.array(trec_dataset['test'][label_col]),
                                 target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.SETFIT:
        train, test = TextDataset(np.array(trec_dataset['train']['text']),
                           np.array(trec_dataset['train'][label_col]),
                           target_labels=np.arange(num_classes)), \
               TextDataset(np.array(trec_dataset['test']['text']),
                           np.array(trec_dataset['test'][label_col]),
                           target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(trec_dataset['train']['text'],
                            trec_dataset['train'][label_col],
                            trec_dataset['test']['text'],
                            trec_dataset['test'][label_col])
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_tps(trec_dataset['train']['text'],
                            trec_dataset['train'][label_col],
                            trec_dataset['test']['text'],
                            trec_dataset['test'][label_col])
    else:
        raise ValueError('Unsupported dataset type for dataset "' + dataset_name + '"')


def _read_agn(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
              classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    agn_dataset = datasets.load_dataset('ag_news')
    num_classes = 4

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        train, test = _text_to_huggingface(tokenizer,
                                           agn_dataset['train']['text'],
                                           agn_dataset['train']['label'],
                                           agn_dataset['test']['text'],
                                           agn_dataset['test']['label'],
                                           num_classes,
                                           int(dataset_kwargs['max_length']))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.RAW:
        train, test = RawDataset(np.array(agn_dataset['train']['text']),
                                 np.array(agn_dataset['train']['label']),
                                 target_labels=np.arange(num_classes)), \
                      RawDataset(np.array(agn_dataset['test']['text']),
                                 np.array(agn_dataset['test']['label']),
                                 target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.SETFIT:
        train, test = TextDataset(np.array(agn_dataset['train']['text']),
                                  np.array(agn_dataset['train']['label']),
                                  target_labels=np.arange(num_classes)), \
                      TextDataset(np.array(agn_dataset['test']['text']),
                                  np.array(agn_dataset['test']['label']),
                                  target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_dbp_140k(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
              classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets

    data_files = {'train': 'train.parquet', 'test': 'test.parquet'}
    dbp_dataset = datasets.load_dataset('./data/dbp-140k', data_files=data_files)
    num_classes = 14

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        train, test = _text_to_huggingface(tokenizer,
                                           dbp_dataset['train']['content'],
                                           dbp_dataset['train']['label'] ,
                                           dbp_dataset['test']['content'],
                                           dbp_dataset['test']['label'],
                                           num_classes,
                                           int(dataset_kwargs['max_length']))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.RAW:
        train, test = RawDataset(np.array(dbp_dataset['train']['content']),
                                 np.array(dbp_dataset['train']['label']),
                                 target_labels=np.arange(num_classes)), \
                      RawDataset(np.array(dbp_dataset['test']['content']),
                                 np.array(dbp_dataset['test']['label']),
                                 target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.SETFIT:
        train, test = TextDataset(np.array(dbp_dataset['train']['content']),
                                  np.array(dbp_dataset['train']['label']),
                                  target_labels=np.arange(num_classes)), \
                      TextDataset(np.array(dbp_dataset['test']['content']),
                                  np.array(dbp_dataset['test']['label']),
                                  target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_imdb(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
                 classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    imdb_dataset = datasets.load_dataset('imdb', ignore_verifications=True)
    num_classes = 2

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        train, test = _text_to_huggingface(tokenizer,
                                           imdb_dataset['train']['text'],
                                           imdb_dataset['train']['label'],
                                           imdb_dataset['test']['text'],
                                           imdb_dataset['test']['label'],
                                           num_classes,
                                           int(dataset_kwargs['max_length']))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.RAW:
        train, test = RawDataset(np.array(imdb_dataset['train']['text']),
                                 np.array(imdb_dataset['train']['label']),
                                 target_labels=np.arange(num_classes)), \
                      RawDataset(np.array(imdb_dataset['test']['text']),
                                 np.array(imdb_dataset['test']['label']),
                                 target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    elif dataset_type == DataSetType.SETFIT:
        train, test = TextDataset(np.array(imdb_dataset['train']['text']),
                                  np.array(imdb_dataset['train']['label']),
                                  target_labels=np.arange(num_classes)), \
                      TextDataset(np.array(imdb_dataset['test']['text']),
                                  np.array(imdb_dataset['test']['label']),
                                  target_labels=np.arange(num_classes))
        train, valid = get_validation_set(train, validation_set_size=0.1)
        return train, valid, test, num_classes
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


DATASET_READERS = dict()
DATASET_READERS[DataSets.AG_NEWS] = _read_agn
DATASET_READERS[DataSets.TREC] = _read_trec
DATASET_READERS[DataSets.DBP_140K] = _read_dbp_140k
DATASET_READERS[DataSets.IMDB] = _read_imdb


def read_dataset(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
                 classifier_kwargs: dict, dataset_type=None, data_dir='.data'):

    dataset = DataSets.from_str(dataset_name)
    if dataset in DATASET_READERS.keys():
        return DATASET_READERS[dataset](dataset_name, dataset_kwargs, classifier_name,
                                        classifier_kwargs, dataset_type, data_dir=data_dir)
    else:
        raise DatasetReaderNotFoundException(f'No reader registered for dataset \'{dataset_name}\'')
