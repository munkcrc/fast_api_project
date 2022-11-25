from __future__ import annotations
from typing import Callable, Hashable, Union, Optional, List, Literal, Sequence
from cr.data.segmentation.segmentation import SegmentationMethod, CompositeSegmentationMethod
import numpy as np


class SourcedArray(np.ndarray):
    def __new__(cls, array, dataset, name):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(array).view(cls)
        # add the new attribute to the created instance
        obj.dataset = dataset
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.dataset = getattr(obj, 'dataset', None)
        self.name = getattr(obj, 'name', None)

class DataSet(object):

    def __init__(self, id, df):
        self._id = id
        self._root_dataframe = df
        self._segmentations = []

    @property
    def id(self):
        return self._id

    @property
    def _df(self):
        return self._root_dataframe

    @property
    def observations(self):
        return self._df.shape[0]

    @property
    def segmentations(self):
        return self._segmentations

    def __getitem__(self, id: Hashable) -> np.ndarray:
        if isinstance(id, (list, tuple)):
            return [SourcedArray(self._df[id_].values, dataset=self, name=id_) for id_ in id]
        return SourcedArray(self._df[id].values, dataset=self, name=id)

    def segment(self, by: str, method: SegmentationMethod) -> Segmentation:
        segmentation = [segmentation for segmentation in self.segmentations if
                        segmentation.by == by and segmentation.method == method]
        if segmentation:
            return segmentation[0]
        # Create a segmentation based on the chosen segmentationMethod
        self._segmentations.append(Segmentation(self, by, method))
        return self._segmentations[-1]

    def composite_segmentations(self, *segmentations, store=False):
        by=[segmentation.by for segmentation in segmentations]
        method=CompositeSegmentationMethod([segmentation.method for segmentation in segmentations])
        if store:
            return self.segment(by, method)
        return Segmentation(self, by, method)

    def iter_all_datasets(self):
        yield self
        for segmentation in self.segmentations:
            for segment in segmentation.segments:
                for dataset in segment.iter_all_datasets():
                    yield dataset

    def iter_all_segmentations(self):
        for dataset in self.iter_all_datasets():
            for segmentation in dataset.segmentations:
                yield segmentation
            
    def __repr__(self):
        return f"{self.id}"

    def __str__(self):
        return f"{self.__repr__()}: {self.observations} observations and {self._df.shape[1]} variables"

class Segment(DataSet):
    def __init__(self, parent, indexes, by, segment_id):
        super().__init__(parent.id, parent._root_dataframe)
        self.parent = parent
        self._indexes = indexes
        self.by = by
        self.segment_id = segment_id

    @property
    def id(self):
        return f"{super().id}>{self.by}={self.segment_id}"

    @property
    def _df(self):
        return self.parent._df.iloc[self._indexes]

class Segmentation(object):
    def __init__(self, root_dataset, by, method):
        # TODO: should a segmentation contain an 'uncovered' in cases where observations fall out of a segmentation?
        self.root_dataset = root_dataset
        self.by = by
        self.method = method
        self._segments = self._create_segments()

    @property
    def approach(self):
        return self.method.__class__.__name__

    @property
    def id(self):
        return f"{self.root_dataset.id}>{self.by}|{self.method}"

    @property
    def segments(self):
        return self._segments

    def composite_with(self, other_segmentation):
        return self.root_dataset.composite_segmentations(self, other_segmentation)

    def _create_segments(self):
        segment_ids, segment_indexes = self.method.segment(self.root_dataset[self.by])
        segments = []
        for key, indexes in zip(segment_ids, segment_indexes):
            segment = Segment(
                parent=self.root_dataset,
                indexes=indexes,
                by=self.by,
                segment_id=key
            )
            segment.segmentation = self
            segments.append(segment)
        return segments

    def __getitem__(self, id: Hashable) -> Segment:
        for segment in self.segments:
            if segment.segment_id == id:
                return segment
    
    def __iter__(self):
        return self.segments.__iter__()

    def __repr__(self):
        return f"<Segmentation: {self.root_dataset.id} using {self.approach} by {self.by}>"