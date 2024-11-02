import torch
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from small_text.data.datasets import SklearnDataset
from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
from small_text.integrations.transformers.datasets import TransformersDataset
from transformers import AutoTokenizer


def _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs):
    tokenizer_name = classifier_kwargs['transformer_tokenizer'] if 'transformer_tokenizer' in classifier_kwargs \
        else classifier_kwargs['transformer_model']
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir='.workspace_cache/',
    )
    return tokenizer
 

def _text_to_huggingface(tokenizer, train_text, train_labels, test_text, test_labels, num_classes, max_length,
                         is_multi_label=False):
    assert is_multi_label is True and isinstance(train_labels, csr_matrix) or \
           is_multi_label is False and isinstance(train_labels, (list, np.ndarray))

    return TransformersDataset.from_arrays(train_text, train_labels, tokenizer,
                                           target_labels=np.arange(num_classes), max_length=max_length), \
        TransformersDataset.from_arrays(test_text, test_labels, tokenizer,
                                        target_labels=np.arange(num_classes), max_length=max_length)


def _text_to_bow(x, y, x_test, y_test, max_features=50000, ngram_range=(1, 2)):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    x = vectorizer.fit_transform(x)
    x_test = vectorizer.transform(x_test)

    return (SklearnDataset(normalize(x), y),
            SklearnDataset(normalize(x_test), y_test))


def _text_to_tps(x_text, y, x_test_text, y_test):
    try:
        from torchtext.legacy import data
    except AttributeError:
        from torchtext import data

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False, unk_token=None, pad_token=None)

    fields = [('text', text_field), ('label', label_field)]

    train = data.Dataset([data.Example.fromlist([text, labels], fields)
                          for text, labels in zip(x_text, y)],
                         fields)
    test = data.Dataset([data.Example.fromlist([text, labels], fields)
                         for text, labels in zip(x_test_text, y_test)],
                        fields)

    text_field.build_vocab(train, min_freq=1)
    label_field.build_vocab(train)

    train_tc = _dataset_to_text_classification_dataset(train)
    test_tc = _dataset_to_text_classification_dataset(test)

    return train_tc, test_tc


def _dataset_to_text_classification_dataset(dataset):
    """

    Parameters
    ----------
    data : torchtext.data.Dataset
    """

    assert dataset.fields['text'].vocab.itos[0] == '<unk>'
    assert dataset.fields['text'].vocab.itos[1] == '<pad>'
    unk_token_idx = 1

    vocab = dataset.fields['text'].vocab

    data = [(torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                               for token in example.text]),
             dataset.fields['label'].vocab.stoi[example.label])
            for example in dataset.examples]

    return PytorchTextClassificationDataset(data, vocab)
