import numpy as np


def label(y_true, num_classes, label_noise=None):
    def get_probabilities(num_classes, y_i):

        if label_noise is not None and label_noise > 0.0:
            p = label_noise / (num_classes - 1)
            arr = np.array([p] * num_classes)
            arr[y_i] = (1 - label_noise)
        else:
            p = 1.0 / (num_classes - 1)
            arr = np.array([p] * num_classes)
            arr[y_i] = 0
        arr = arr / arr.sum()
        return arr

    if label_noise is None or label_noise == 0.0:
        return y_true
    else:
        print(f'True Labels: {y_true}')
        print(f'Noisy labels: {np.array([np.random.choice(np.arange(num_classes, dtype=int), size=1, p=get_probabilities(num_classes, yy)) for yy in y_true]).flatten()}')
        return np.array([
            np.random.choice(np.arange(num_classes, dtype=int), size=1, p=get_probabilities(num_classes, yy))
            for yy in y_true
        ]).flatten()
