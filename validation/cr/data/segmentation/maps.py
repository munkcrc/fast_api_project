from .utilities import _repr, _str, _eq
from typing import List, Tuple
import numpy as np

class SegmentationMap(object):
    """ A SegmentationMap can map observations into predefined segments """
    
    def __init__(self, pre_map:List = None):
        if not pre_map:
            pre_map = []
        self.pre_map = pre_map

    def transform_pre_map(self, values):
        for transformation in self.pre_map:
            values = transformation.transform(values)
        return values

    def segment(self, values:np.ndarray):
        values = self.transform_pre_map(values)
        segment_ids = []
        segment_indexes = []
        unsegmented = values.shape[0]
        for in_segment, group_id in self._iter_segment(values):
            segment_ids.append(group_id)
            segment_indexes.append(np.where(in_segment))
            unsegmented -= np.sum(in_segment)

        if unsegmented != 0:
            # TODO: Throw a warning?
            pass 
        return segment_ids, segment_indexes
    
    def to_dict(self):
        return dict(pre_map=self.pre_map)

class MapByGroups(SegmentationMap):

    def __init__(self, groups:List):
        super().__init__()
        self.groups = groups

    def __eq__(self, other):
        return _eq(self, other, ['groups', 'pre_map'])

    def __repr__(self):
        return _repr(self, ['groups', 'pre_map'])

    def __str__(self):
        return _str(self, ['groups', 'pre_map'])

    def _iter_segment(self, values:np.ndarray) -> Tuple[np.ndarray, str]:
        #  TODO: Assert we have no duplicates in groups
        #        we should atleast raise a warning if we have a obs in multiple segments
        for group in self.groups:
            if isinstance(group, list):
                yield np.isin(values,group), group
            else: 
                yield values==group, group

    def to_dict(self):
        return {**super().to_dict(), "groups": list(self.groups)}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict["groups"])
        obj.pre_map = dict["pre_map"]
        return obj

class MapByBins(SegmentationMap):

    def __init__(self, bins:List[float]):
        super().__init__()
        self.bins = bins

    def __eq__(self, other):
        return _eq(self, other, ['bins', 'pre_map'])

    def __repr__(self):
        return _repr(self, ['bins', 'pre_map'])

    def __str__(self):
        return _str(self, ['bins', 'pre_map'])

    def _iter_segment(self, values:np.ndarray) -> Tuple[np.ndarray, str]:
        # increase last upper boundary in bins to ensure all values are captured in a
        # bin. This ensures that bins[-2] <= x <= bins[-1] instead of
        # bins[-2] <= x < bins[-1]
        act_bins = [-np.inf] + self.bins + [np.inf]
        observation_segment = np.digitize(values, act_bins) - 1

        for i, (segment_start, segment_end) in enumerate(zip(act_bins[:-1], act_bins[1:])):
            yield observation_segment == i, f"[{segment_start}, {segment_end})"

    def to_dict(self):
        return {**super().to_dict(), "bins": list(self.bins)}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict["bins"])
        obj.pre_map = dict["pre_map"]
        return obj

class MapDatesByBins(SegmentationMap):

    def __init__(self, bins: List[str], time_unit: str):
        super().__init__()
        self.bins = bins
        self.time_unit = time_unit

    def __eq__(self, other):
        return _eq(self, other, ['bins', 'time_unit', 'pre_map'])

    def __repr__(self):
        return _repr(self, ['bins', 'time_unit', 'pre_map'])

    def __str__(self):
        return _str(self, ['bins', 'time_unit', 'pre_map'])

    def _iter_segment(self, values:np.ndarray) -> Tuple[np.ndarray, str]:
        # increase last upper boundary in bins to ensure all values are captured in a
        # bin. This ensures that bins[-2] <= x <= bins[-1] instead of
        # bins[-2] <= x < bins[-1]
        values_as_int = values.astype(f'datetime64[{self.time_unit}]').astype(int)
        bins_as_int = [
            np.datetime64(elem, self.time_unit).astype(int) for elem in self.bins]
        act_bins_as_int = [-np.inf] + bins_as_int + [np.inf]
        observation_segment = np.digitize(values_as_int, act_bins_as_int) - 1
        date_min = str(np.datetime64('0001-01-01 00:00:00.000000000', self.time_unit))
        date_max = str(np.datetime64('9999-12-31 23:59:59.999999999', self.time_unit))
        act_bins = [date_min] + self.bins + [date_max]

        for i, (segment_start, segment_end) in enumerate(zip(act_bins[:-1], act_bins[1:])):
            yield observation_segment == i, f"[{segment_start}, {segment_end})"

    def to_dict(self):
        return {**super().to_dict(), "bins": list(self.bins),
                "time_unit": str(self.time_unit)}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict["bins"], dict["time_unit"])
        obj.pre_map = dict["pre_map"]
        return obj

