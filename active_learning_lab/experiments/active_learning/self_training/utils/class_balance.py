import numpy as np

# COPIED from ClassBalancer query strategy
def _query_class_balanced(y_pred, scores, num_classes, y, n):
    ignored_classes = []
    target_distribution = _get_rebalancing_distribution(n,
                                                        num_classes,
                                                        y,
                                                        y_pred,
                                                        ignored_classes=ignored_classes)

    active_classes = np.array([i for i in range(num_classes) if i not in set(ignored_classes)])

    indices_balanced = []
    for c in active_classes:
        class_indices = np.argwhere(y_pred == c)[:, 0]
        if target_distribution[c] > 0:
            p = scores[class_indices] / scores[class_indices].sum()
            if np.all(p >= 0) and np.all(p <= 1):
                queried_indices = np.random.choice(class_indices, target_distribution[c], replace=False, p=p)
                indices_balanced.extend(queried_indices.tolist())
            else:
                raise AssertionError

    return np.array(indices_balanced)


# <COPIED from ClassBalancer query strategy>
# The naming here is kept general (distributions and categories), but this is currently used to create distributions
#  over the number of classes
from small_text.data.sampling import _get_class_histogram
def _sample_distribution(num_samples: int,
                         source_distribution,
                         ignored_values):
    """Return a balanced sample from the given `source_distribution` of size ``num-samples`. The sample is represented
    in the form of an empirical categorial frequency distribution (i.e. a histogram). It is built iteratively, and
    prefers the category currently having the smallest number of samples.

    Parameters
    ----------
    num_samples : int
        Number of samples that the resulting distribution has.
    source_distribution : np.ndarray[int]
        A source frequency distribution in the shape of (num_values,) where num_values is the number of possible values
        for the source_distribution.
    ignored_values : list of int
        List of values (indices in the interval [0, `source_distribution.shape[0]`]) that should be ignored.

    Returns
    -------
    output_distribution : np.ndarray[int]
        A new distribution, which is  whose categories are less than or equal to the source distribution.
    """

    num_classes = source_distribution.shape[0]
    active_classes = np.array([i for i in range(num_classes) if i not in set(ignored_values)])

    new_distribution = np.zeros((num_classes,), dtype=int)
    for _ in range(num_samples):
        distribution_difference = (new_distribution - source_distribution)[active_classes]
        minima = np.where(distribution_difference == distribution_difference.min())[0]

        # Sample the class which occurs the least. In the case of a tie, the decision is random.
        if minima.shape[0] == 1:
            new_distribution[active_classes[minima[0]]] += 1
        else:
            sampled_minimum_index = np.random.choice(minima, 1)[0]
            new_distribution[active_classes[sampled_minimum_index]] += 1

    return new_distribution


def _get_rebalancing_distribution(num_samples, num_classes, y, y_pred, ignored_classes=[]):
    current_class_distribution = _get_class_histogram(y, num_classes)
    predicted_class_distribution = _get_class_histogram(y_pred, num_classes)

    number_per_class_required_for_balanced_dist = current_class_distribution.max() - current_class_distribution

    number_per_class_required_for_balanced_dist[list(ignored_classes)] = 0

    # Balancing distribution: When added to current_class_distribution, the result is balanced.
    optimal_balancing_distribution = current_class_distribution.max() - current_class_distribution
    target_distribution = _sample_distribution(num_samples,
                                               optimal_balancing_distribution,
                                               ignored_values=ignored_classes)

    # balancing_distribution:
    balancing_distribution = np.zeros((num_classes,), dtype=int)
    active_classes = np.array([i for i in range(num_classes) if i not in set(ignored_classes)])

    for c in active_classes:
        if predicted_class_distribution[c] < target_distribution[c]:
            # adapt the balancing distribution so that it can be sampled
            balancing_distribution[c] = predicted_class_distribution[c]
        else:
            balancing_distribution[c] = target_distribution[c]

    # The predicted labels does not have enough classes so that a sample with the desired balancing distribution
    # cannot be provided. Try to fill the remainder with other samples from "active classes" instead.
    remainder = target_distribution.sum() - balancing_distribution.sum()
    if remainder > 0:
        current_class_distribution += balancing_distribution

        free_active_class_samples = []
        for c in active_classes:
            class_indices = np.argwhere(y_pred == c)[:, 0]
            if class_indices.shape[0] > current_class_distribution[c]:
                free_active_class_samples.extend([c] * (class_indices.shape[0] - current_class_distribution[c]))

        np.random.shuffle(free_active_class_samples)
        for c in free_active_class_samples[:remainder]:
            balancing_distribution[c] += 1
            current_class_distribution[c] += 1

    # When not enough samples can be taken from the active classes, we fall back to using all classes.
    remainder = target_distribution.sum() - balancing_distribution.sum()
    if remainder > 0:
        free_ignored_class_samples = []
        for i, count in enumerate(predicted_class_distribution - balancing_distribution):
            if count > 0:
                free_ignored_class_samples.extend([i] * count)

        np.random.shuffle(free_ignored_class_samples)
        for c in free_ignored_class_samples[:remainder]:
            balancing_distribution[c] += 1

    return balancing_distribution
# </COPIED from ClassBalancer query strategy>
