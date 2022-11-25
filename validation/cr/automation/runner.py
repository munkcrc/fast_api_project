from datetime import date
from importlib import import_module
from pathlib import Path
from typing import List, Union
from functools import partial
import yaml

from cr.data.segmentation.segmentation import SegmentationMethod
from .recording import is_recording
from .taper import Tape

class Runner():
    def __init__(self, tape:Union[dict, str, Path, Tape] , current_datasets:dict=None):
        if not current_datasets:
            current_datasets = {}

        self.datasets = current_datasets

        if isinstance(tape, Tape):
            # If we are getting a tape with stored objects
            # We make sure to copy the tests over directly
            # Such that run can pull the results directly
            if tape.has_stored_objects:
                self._runs = tape.objects
            tape = tape.to_dict()
        elif not isinstance(tape, dict):
            with open(tape, 'rb') as file:
                tape = yaml.safe_load(file)

        # If self._runs we are using a stored tape
        # and it makes no sense to set tests, datasets etc.
        self.tests = tape['tests']
        self.dataset_definitions = tape['datasets']
        
        if not hasattr(self, "_runs"):
            self._runs = {}

    def _run_callable(self, definition, recording_uuid=None, dry_run=False):
        func = self._deserialized_function(definition)
        args = self.get_args(definition.pop('args', []))
        kwargs = self.get_kwargs(definition.pop('kwargs', {}))

        # If we are recording, make sure we maintain the recording_uuid
        if is_recording():
            kwargs['_dry_run'] = dry_run
            if recording_uuid:
                kwargs['recording_uuid'] = recording_uuid
        
        return func(*args, **kwargs)

    def run(self, uid:str, dry_run:bool=False):
        if uid in self._runs and not dry_run:
            return self._runs[uid]

        self._runs[uid] = self._run_callable(self.tests[uid].copy(), recording_uuid=uid, dry_run=dry_run)
        return self._runs[uid]

    def _deserialize_definition(self, definition):
        if isinstance(definition, list):
            return [self._deserialize_definition(item) for item in definition]
        if isinstance(definition, dict):
            if 'cr_type' not in definition:
                return {key:self._deserialize_definition(value) for key, value in definition.items()}
            if definition['cr_type'] == 'dataset':
                return self.get_dataset(definition['dataset'])
            if definition['cr_type'] == 'sourcedarray':
                dataset = self.get_dataset(definition['dataset'])
                return dataset[definition['name']]
            if definition['cr_type'] == 'class':
                return self._init_deserialized_class(definition)
            if definition['cr_type'] == 'function':
                return self._deserialized_function(definition)
            if definition['cr_type'] == 'partial':
                return partial(
                    self._deserialized_function(definition['function']),
                    *self._deserialize_definition(definition['args']),
                    **self._deserialize_definition(definition['keywords']))
            if definition['cr_type'] == 'result':
                return self.run(definition['source_uid'])

        return definition

    def _init_deserialized_class(self, definition):
        # Only classes which suppert from_dict*
        module = import_module(definition['module'])
        cls = getattr(module, definition['name'])
        dict_args = self.get_kwargs(definition['dict'])
        return cls.from_dict(dict_args)

    def _deserialized_function(self, definition):
        module = import_module(definition['module'])
        func = getattr(module, definition['name'])
        return func

    def get_args(self, args):
        return [self._deserialize_definition(arg) for arg in args]

    def get_kwargs(self, kwargs):
        return {key:self._deserialize_definition(value) for key, value in kwargs.items()}

    def get_dataset(self, dataset_id):
        if dataset_id not in self.datasets:
            datasets = self.recreate_datasets(dataset_id)
            definition = self.dataset_definitions[dataset_id]
            
            # If it is a segment, we need to ensure we target the parent id
            # as per the taped data
            if 'parent' in definition:
                parent_key = definition['parent']
                parent_id = self.datasets[parent_key].id
                for dataset in datasets:
                    self.datasets[dataset.id.replace(parent_id, parent_key)] = dataset
            else:
                for dataset in datasets:
                    self.datasets[dataset.id] = dataset
        return self.datasets[dataset_id]

    def recreate_datasets(self, dataset_id):
        # The dataset is already instantiated
        if dataset_id in self.datasets:
            return [self.datasets[dataset_id]]

        # Find the dataset definition
        definition = self.dataset_definitions[dataset_id]
        
        # If it's a source dataset ingest it
        if isinstance(definition['source'], dict):
            return [self.ingest_dataset(definition, dataset_id)]

        # Otherwise get the parent and resolve the Segmentation
        parent = self.get_dataset(definition['parent'])

        segmentation_definition = definition['segmentation']
        
        segmentation_method = self._init_deserialized_class(segmentation_definition['method'])

        return parent.segment(segmentation_definition['by'], segmentation_method)

    def ingest_dataset(self, definition, id_):
        dataset = self._run_callable(definition['source'].copy())
        # TODO: A bit dirty - is there an alternative way to ensure ID is maintained?
        dataset._id = id_
        return dataset

    def get_run_context(self):
        return {
            "_DATE": date.today().strftime('%B %d, %Y'),
            "_ID": "ABCDEFG"
        }