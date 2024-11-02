import torch
import numpy as np

from pathlib import Path

from gensim.models.word2vec import Word2VecKeyedVectors


def get_embedding_matrix(name, vocab, data_dir='.data/'):

    embedding_dir = Path(data_dir).joinpath('embeddings')
    embedding_dir.mkdir(parents=True, exist_ok=True)

    serialized_file = embedding_dir.joinpath(name + '.bin')
    if not serialized_file.exists():
        if name == 'word2vec':
            model = _load_word2vec(vocab)
            model.save(str(serialized_file.resolve()))
            return _build_embedding_matrix_from_keyedvectors(model, vocab)
        elif name == 'glove':
            return _load_flair_embedding('glove', vocab)
    else:
        if name == 'word2vec':
            model = Word2VecKeyedVectors.load(str(serialized_file.resolve()), mmap='r')
            return _build_embedding_matrix_from_keyedvectors(model, vocab)
        elif name == 'glove':
            return _load_flair_embedding('glove', vocab)

    raise ValueError(f'Unknown embedding: "{name}"')


def _load_word2vec(vocab, min_freq=1):
    #assert vocab.itos[0] == '<pad>'
    import gensim.downloader as api
    return api.load('word2vec-google-news-300')


def _build_embedding_matrix_from_keyedvectors(pretrained_vectors, vocab, min_freq=1):

    vectors = [
        np.zeros(pretrained_vectors.vectors.shape[1])  # <pad>
    ]
    num_special_vectors = len(vectors)
    vectors += [
        pretrained_vectors.vectors[pretrained_vectors.vocab[vocab.itos[i]].index]
        if vocab.itos[i] in pretrained_vectors.vocab
        else np.zeros(pretrained_vectors.vectors.shape[1])
        ##np.random.uniform(-0.01, 0.01, precomputed_glove.vectors.shape[1])
        for i in range(num_special_vectors, len(vocab))
    ]
    for i in range(num_special_vectors, len(vocab)):
        if vocab.itos[i] not in pretrained_vectors.vocab and vocab.freqs[vocab.itos[i]] >= min_freq:
            vectors[i] = np.random.uniform(-0.25, 0.25, pretrained_vectors.vectors.shape[1])

    return torch.as_tensor(np.stack(vectors))


def _load_flair_embedding(name, vocab):
    from flair.embeddings import WordEmbeddings

    assert vocab.stoi['<ukn>'] == 0
    assert vocab.stoi['<pad>'] == 1

    glove_embedding = WordEmbeddings(name)

    precomputed_glove = glove_embedding.precomputed_word_embeddings
    precomputed_vocab = glove_embedding.precomputed_word_embeddings.vocab
    precomputed_vectors = glove_embedding.precomputed_word_embeddings.vectors

    # missing_vector = np.random.uniform(-0.25, 0.25, precomputed_glove.vectors.shape[1])

    vectors = [
        np.random.uniform(-0.16, 0.16, precomputed_glove.vectors.shape[1]),  # <ukn>
        np.zeros(precomputed_glove.vectors.shape[1])  # <pad>
    ]
    vectors += [
        precomputed_vectors[precomputed_vocab[vocab.itos[i]].index]
        if vocab.itos[i] in precomputed_glove.vocab
        else np.zeros(precomputed_glove.vectors.shape[1])
        ##np.random.uniform(-0.01, 0.01, precomputed_glove.vectors.shape[1])
        for i in range(2, len(vocab))
    ]
    for i in range(2, len(vocab)):
        if vocab.itos[i] not in precomputed_glove.vocab and vocab.freqs[vocab.itos[i]] >= 2:
            vectors[i] = np.random.uniform(-0.16, 0.16, precomputed_glove.vectors.shape[1])

    return torch.as_tensor(np.stack(vectors))
