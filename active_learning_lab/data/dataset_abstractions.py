import numpy as np

from small_text.data.datasets import SklearnDataset


class RawDataset(SklearnDataset):

    def __init__(self, x, y, is_multi_label=False, target_labels=None):
        super().__init__(x, y, target_labels=target_labels)
        self.x = np.array(x)
        self.multi_label = is_multi_label

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
        if self.track_target_labels:
            self._infer_target_labels(self._y)

    @property
    def target_labels(self):
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        self._target_labels = target_labels

    def __getitem__(self, item):
        if self.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(self._target_labels)

        if isinstance(item, list) or isinstance(item, np.ndarray) or isinstance(item, slice):
            return RawDataset(np.array(self._x[item]),
                              self._y[item],
                              target_labels=target_labels)

        ds = RawDataset(self._x[item],
                        self._y[item],
                        target_labels=target_labels)
        if len(ds._x.shape) <= 1:
            ds._x = np.expand_dims(ds._x, axis=0)
            ds._y = np.expand_dims(ds._y, axis=0)

        return ds

    def __iter__(self):
        for i in range(self._x.shape[0]):
            yield self[i]

    def __len__(self):
        return self._x.shape[0]
