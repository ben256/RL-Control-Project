import numpy as np


class Box:
    def __init__(self, low, high, shape=None):
        self.low = low
        self.high = high

        if shape is None:
            assert (
                    low.shape == high.shape
            ), "Low and High must have the same shape"
            self._shape = low.shape
        else:
            self._shape = shape

    @property
    def shape(self):
        return self._shape

    def is_bounded(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)

    def bounds(self):
        return self.low, self.high