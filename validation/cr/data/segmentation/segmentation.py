from typing import Callable, Dict, List, Tuple
import numpy as np
import itertools
from functools import reduce, partial
from .maps import SegmentationMap

class SegmentationMethod(object):  
    """ A SegmentationMethod can both define segments and map observations into these segments"""

    def compute_map(self, values:np.ndarray) -> SegmentationMap:
        raise NotImplementedError("This SegmentationMethod failed to implement a map computation")

    def segment(self, values:np.ndarray) -> Dict:
        # If the map is already instantiated or always recompute
        if not hasattr(self, 'map') or self._always_recompute:
            self.map = self.compute_map(values)
        return self.map.segment(values)

    @property
    def _always_recompute(self):
        # If a segmentationmethod should always recompute it map!
        return False

    def to_dict(self):
        return dict(map = self.map.to_dict())

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def from_dict(cls, dict):
        raise NotImplementedError(f"{cls.__name__} does not implement from_dict")

class CompositeSegmentationMethod(SegmentationMethod):

    def __init__(self, methods:List[SegmentationMethod]):
        self.methods = methods

    def segment(self, values:np.ndarray) -> Dict:
        # Take the cartesian product of the segmentation methods 
        # each methods return [(segment_id, segment_indexes), ...]
        # we merge these such that we, for methods A and B iterate all combinations of:
        # [[(A_segment_id, A_segment_indexes),(B_segment_id, B_segment_indexes)], ...]
        indexes = []
        keys = []
        for cross_segmentation in itertools.product(*[zip(*method.segment(x)) for method, x in zip(self.methods, values)]):
            keys.append([segment[0] for segment in cross_segmentation])
            # Get all the values that are in ALL the segments
            indexes.append(reduce(partial(np.intersect1d, assume_unique=True),[segment[1] for segment in cross_segmentation]))
        return keys, indexes

    def to_dict(self):
        return dict(methods=self.methods)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def __repr__(self):
        return ",".join([method.__repr__() for method in self.methods])