import numpy as np

from scipy.sparse import csr_matrix


from small_text.data.sampling import multilabel_stratified_subsets_sampling

from small_text.integrations.pytorch.utils.data import _get_class_histogram
from small_text.data.sampling import balanced_sampling, stratified_sampling


def get_class_histogram(y, num_classes, normalize=True):
    ind, counts = np.unique(y, return_counts=True)
    ind_set = set(ind)

    histogram = np.zeros(num_classes)
    for i, c in zip(ind, counts):
        if i in ind_set:
            histogram[i] = c

    if normalize:
        return histogram / histogram.sum()

    return histogram.astype(int)


def get_validation_set(dataset, strategy='balanced', validation_set_size=0.1,
                       multilabel_strategy='labelsets'):

    if validation_set_size == 0.0:
        return None

    n_samples = int(validation_set_size * len(dataset))
    y = dataset.y

    if strategy == 'balanced':
        # draw a random subset in the size of the half dataset first to prevent that the balancing
        # mechanism uses all labels of a rare class
        indices_random_subset = np.random.choice(np.arange(y.shape[0]), size=np.floor(y.shape[0]//2).astype(int), replace=False)
        indices_sampled = balanced_sampling(y[indices_random_subset], n_samples=n_samples)
        return _train_valid(dataset, indices_random_subset[indices_sampled])
    elif strategy == 'random':
        indices = np.random.permutation(len(dataset))
        indices_sampled = indices[:n_samples]
        return _train_valid(dataset, indices_sampled)
    elif strategy == 'stratified':
        if isinstance(y, csr_matrix):
            if multilabel_strategy == 'labelsets':
                indices_sampled = multilabel_stratified_subsets_sampling(y, n_samples=n_samples)
            else:
                raise ValueError(f'Invalid multilabel_strategy: {multilabel_strategy}')
        else:
            indices_sampled = stratified_sampling(y, n_samples=n_samples)
        return _train_valid(dataset, indices_sampled)

    raise ValueError(f'Invalid strategy: {strategy}')


def _train_valid(dataset, sampled_indices):
    indices_all = np.arange(len(dataset))
    mask = np.isin(indices_all, sampled_indices)
    return dataset[indices_all[~mask]].clone(), dataset[indices_all[mask]].clone()


def get_class_weights(y, num_classes, eps=1e-8):
    label_counter = _get_class_histogram(y, num_classes, normalize=False)
    pos_weight = np.ones(num_classes, dtype=float)
    num_samples = len(y)
    for c in range(num_classes):
        pos_weight[c] = (num_samples - label_counter[c]) / (label_counter[c] + eps)

    if num_classes == 2:
        pos_weight[pos_weight.argmin()] = 1.0

    return pos_weight
