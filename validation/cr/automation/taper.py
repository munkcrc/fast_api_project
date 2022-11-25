from typing import Callable
from cr.data.dataset import DataSet, Segment, SourcedArray, Segmentation
from cr import __version__
from datetime import datetime
from functools import partial
from types import FunctionType
import yaml
import numpy as np

from cr.testing.result import Result

class Tape(object):

    def __init__(self, store_objects=False):
        self._store_objects = store_objects
        self.objects = {} # only used as an intermediate cache
        self.tests = {}
        self.datasets = {}
        self.meta = dict(
            cr = __version__,
            date = datetime.today().strftime("%d/%m/%Y")
        )

    @property
    def has_stored_objects(self):
        return self._store_objects

    def to_yaml(self, stream=None):
        return yaml.dump(self.to_dict(), stream)

    def to_dict(self):
        return {
            'datasets': self.datasets, 
            'tests': self.tests, 
            'meta': self.meta
        }

    def record_test(self, func:Callable, args:list, kwargs:dict, uid:str, result):
        entry = dict(
            module = func.__module__,
            name = func.__name__
        )
        if kwargs:
            entry['kwargs'] = self._serialize_object(kwargs)
        if args:
            entry['args'] = self._serialize_object(args)
        
        self.tests[uid] = entry

        if self._store_objects:
            self.objects[uid] = result

    def record_ingestion(self, func:Callable, args:list, kwargs:dict, dataset:DataSet):
        entry = dict(
            module = func.__module__,
            name = func.__name__
        )
        if kwargs:
            entry['kwargs'] = self._serialize_object(kwargs)
        if args:
            entry['args'] = self._serialize_object(args)
        
        self._add_dataset(dataset, entry)

    def _serialize_segmentation(self, segmentation):
        seg = dict(
            by = segmentation.by,
            method = self._serialize_object(segmentation.method)
        )
        return seg

    def _add_dataset(self, dataset, source=None):
        if not dataset.id in self.datasets:
            if isinstance(dataset, Segment):
                self._add_dataset(dataset.parent)
                self.datasets[dataset.id] = dict(
                    source = "Segment",
                    parent = dataset.parent.id,
                    segment = self._serialize_object(dataset.segment_id),
                    segmentation = self._serialize_segmentation(dataset.segmentation)
                )
            else:
                if source:
                    source = source
                else:
                    source = "unknown"
                self.datasets[dataset.id] = dict(
                    source = source,
                    name = dataset.id
                )
            
            if self._store_objects:
                self.objects[dataset.id] = dataset

    def _serialize_object(self, object):    
        # Serialization of scalars
        if type(object) in (str, float, bool, int):
            return object
        if isinstance(object, np.number):
            return object.item() 
        if isinstance(object, np.str_):
            return object.item() 

        # Serialization of collections (excl - numpy arrays)
        if isinstance(object, list) or isinstance(object, tuple):
            return [self._serialize_object(obj) for obj in object]
        if isinstance(object, dict):
            return {key: self._serialize_object(value) for key, value in object.items()}
        if isinstance(object, set):
            return set([self._serialize_object(obj) for obj in object])

        # Serialization of objects
        if isinstance(object, DataSet):
            self._add_dataset(object)
            return dict(
                cr_type = "dataset",
                dataset = object.id
            )
        if isinstance(object, SourcedArray):
            self._add_dataset(object.dataset)
            return dict(
                cr_type = "sourcedarray",
                dataset = object.dataset.id,
                name = object.name
            )
        if isinstance(object, Segmentation):
            return dict(
                cr_type = "segmentation",
                segments = [
                    self._serialize_object(segment)
                    for segment in object.segments
                ]
            )
        if isinstance(object, np.ndarray):
            return self._serialize_object(list(object))

        if hasattr(object, "to_dict") and hasattr(object, "from_dict"):
            return dict(
                cr_type = "class",
                module = object.__module__,
                name = object.__class__.__name__,
                dict = self._serialize_object(object.to_dict())
            )

        if isinstance(object, FunctionType):
            func_name = object.__name__
            if func_name == 'record_func':
                return self._serialize_object(object._recorded_func)
            if func_name == 'doc_wrapper':
                return self._serialize_object(object._wrapped_func)
            elif func_name == '<lambda>':
                raise ValueError(
                    f"Tried serializing {object}, but unable to serialize a lambda function")
            return dict(
                cr_type="function",
                module=object.__module__,
                name=func_name
            )

        if isinstance(object, partial):
            return dict(
                cr_type="partial",
                function=self._serialize_object(object.func),
                args=self._serialize_object(object.args),
                keywords=self._serialize_object(object.keywords))

        if isinstance(object, Result):
            # TODO: What to do if the result has no _recording_uid?
            return dict(
                cr_type = "result",
                source_uid = object._recording_uid
            )

        if object is None:
            return ""

        raise ValueError(f"Tried serializing {object}, but unable to serialize its type {type(object)}")