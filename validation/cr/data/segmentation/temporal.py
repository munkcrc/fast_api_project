from .segmentation import SegmentationMethod
from .maps import MapByGroups
from .utilities import _repr, _str, _eq
import pandas as pd
import numpy as np

class TemporalTransformation():

    def __init__(self, frequency):
        self.frequency = frequency.casefold()

    def __eq__(self, other):
        return _eq(self, other, ['frequency'])

    def __repr__(self):
        return _repr(self, ['frequency'])

    def __str__(self):
        return _str(self, ['frequency'])

    def transform(self, values):
        values = pd.Series(values)

        if self.frequency == "yearly":
            segmentation_values = values.dt.year.values
        elif self.frequency == "monthly":
            segmentation_values = values.dt.date.astype(str).str[:7]
        elif self.frequency == "quarterly":
            segmentation_values = values.dt.year.map(str).values + " Q" + values.dt.quarter.map(str).values
        elif self.frequency == "month":
            segmentation_values = values.dt.month_name().values
        elif self.frequency == "quarter":
            segmentation_values = "Q" + values.dt.quarter.map(str).values
        else:
            raise ValueError(f"Unknown temporal frequency {self.frequency}")
        return segmentation_values

    def to_dict(self):
        return {"frequency": self.frequency}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict['frequency'])
        return obj

class Temporal(SegmentationMethod):

    def __init__(self, frequency):
        self.frequency = frequency

    def __eq__(self, other):
        return _eq(self, other, ['frequency'])

    def __repr__(self):
        return _repr(self, ['frequency'])

    def __str__(self):
        return _str(self, ['frequency'])

    def compute_map(self, values):
        transformation = TemporalTransformation(self.frequency)
        map = MapByGroups(np.unique(transformation.transform(values)))
        map.pre_map.append(transformation)
        return map

    @property
    def _always_recompute(self):
        return True 

    def to_dict(self):
        return {**super().to_dict(), **{"frequency": self.frequency}}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict['frequency'])
        return obj
