import numpy as np
from .segmentation.utilities import _repr, _str, _eq

class NaNHandler(object):

    def __init__(self, method):
        self.method = method

    def __eq__(self, other):
        return _eq(self, other, ['method'])

    def __repr__(self):
        return _repr(self, ['method'])

    def __str__(self):
        return _str(self, ['method'])

    def transform(self, values):
        nans = np.isnan(values)
        if np.any(nans):
            nans_idx = np.nonzero(nans)
            if self.method == "min":
                values[nans_idx] = np.min(values[~nans])
            elif self.method == "max":
                values[nans_idx] = np.max(values[~nans])
            else:
                values[nans_idx] = self.method
        return values

    def to_dict(self):
        return {"method": self.method}

    @classmethod
    def from_dict(cls, dict):
        return cls(dict["method"])
