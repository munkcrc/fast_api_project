import numpy as np

from .segmentation import SegmentationMethod
from .maps import MapByGroups
from .utilities import _repr, _str, _eq


class ByGroup(SegmentationMethod):

    def __init__(self, groups=None):
        self.groups = groups

    def __eq__(self, other):
        return _eq(self, other, ['groups'])

    def __repr__(self):
        return _repr(self, ['groups'])

    def __str__(self):
        return _str(self, ['groups'])

    def compute_map(self, values):
        if not self.groups:
            # TODO: should this logic be somewhere else?
            if np.issubdtype(values.dtype, object):
                groups = np.unique(values.tolist())
            else:
                groups = np.unique(values)

        else:
            groups = self.groups

        return MapByGroups(groups)

    def to_dict(self):
        return {**super().to_dict(), **{"groups": self.groups}}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict['groups'])
        obj.map = MapByGroups.from_dict(dict["map"])
        return obj