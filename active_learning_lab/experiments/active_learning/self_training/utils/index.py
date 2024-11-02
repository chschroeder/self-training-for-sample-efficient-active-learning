import faiss
import numpy as np

from scipy.special import rel_entr

from small_text.data.sampling import _get_class_histogram


def _initialize_index(embeddings, embeddings_train):
    index = faiss.index_factory(embeddings.shape[1], 'Flat')

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.train(embeddings_train)
    index.add(embeddings)

    return index


def _get_kl_scores(index, indices_subsampled, embeddings, y_pred_proba, y_pred_proba_labeled,
                   mini_batch_size=32, knn_k=7, divergence_beta=0.1):
    num_batches = int(np.ceil(indices_subsampled.shape[0] / mini_batch_size))
    offset = 0

    scores = np.empty((indices_subsampled.shape[0],))

    for batch_idx in np.array_split(np.arange(indices_subsampled.shape[0]), num_batches, axis=0):
        _, indices_nn = index.search(embeddings, knn_k)

        kl_divs_unl = np.apply_along_axis(
            lambda v: np.mean([
                rel_entr(y_pred_proba[v], y_pred_proba_labeled[i])
                for i in indices_nn[v - offset]]),
            0,
            batch_idx[None, :])

        def upper_diag_indices(v):
            return [
                [i, j] for i in range(0, v.shape[0])
                for j in range(i + 1, v.shape[0])
            ]

        kl_divs_lab = np.apply_along_axis(
            lambda v: np.mean([
                rel_entr(y_pred_proba_labeled[v[i]],
                         y_pred_proba_labeled[v[j]])
                for i, j in upper_diag_indices(v)]),
            1,
            indices_nn[offset:(offset + batch_idx.shape[0]), 1:])

        scores[offset:(offset + batch_idx.shape[0])] = kl_divs_unl + divergence_beta * kl_divs_lab
        offset += batch_idx.shape[0]

    return scores


def _get_hard_label_agreement_scores(index, indices_subsampled, embeddings, y_pred, num_classes,
                                     mini_batch_size=128, knn_k=15):
    num_batches = int(np.ceil(indices_subsampled.shape[0] / mini_batch_size))
    offset = 0

    scores = np.empty((indices_subsampled.shape[0],))
    knn_pred = np.empty((indices_subsampled.shape[0],), dtype=int)

    _, indices_nn = index.search(embeddings, knn_k)

    for batch_idx in np.array_split(np.arange(indices_subsampled.shape[0]), num_batches, axis=0):

        hist = np.apply_along_axis(
            lambda v: _get_class_histogram(y_pred[v], num_classes, normalize=True),
            1,
            indices_nn[offset:(offset + batch_idx.shape[0]), 1:])

        scores[offset:(offset + batch_idx.shape[0])] = hist.max(axis=1)
        knn_pred[offset:(offset + batch_idx.shape[0])] = hist.argmax(axis=1)
        offset += batch_idx.shape[0]

    return scores, knn_pred
