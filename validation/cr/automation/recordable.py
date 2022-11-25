from cr.data.dataset import DataSet
from .recording import get_active_tape
from uuid import uuid4

from cr.testing.result import Result, MockResult

# global macro variable True if we are inside a recordable function being called.
_is_recording_caller = False

def recordable(func):
    def record_func(*args, **kwargs):
        global _is_recording_caller
        # If the function is called by some func under recording we don't want to store it
        if _is_recording_caller:
            return func(*args, **kwargs)

        # Get the tape, if there is no tape we are not actively recording, so just call func
        tape = get_active_tape()
        if not tape:
            return func(*args, **kwargs)

        # We have a tape and have to record this function, first get the uid 
        # the uid can be passed as an kwargs to the function alternatively generate.
        # If the result is a dataset the UID is unused - as that is given the dataset.id
        # TODO: Replace uuid4 with something that is more 'communicateable' so we,
        #       can for instance embed in plots - making these easily rerunable without copy-paste.
        uid = kwargs.pop('recording_uuid', str(uuid4()))

        # Run the function and store the result
        try:
            _is_recording_caller = True
            # Are we in a dryrun?
            dry_run = kwargs.pop('_dry_run', False)
            if dry_run:
                result = MockResult()
            else:
                result = func(*args, **kwargs)
        finally:
            _is_recording_caller = False

        # If the result is a Result as expected record the result otherwise fail
        if isinstance(result, Result):
            try:
                tape.record_test(func, args, kwargs, uid=uid, result=result)
            except Exception as err:
                raise err
        elif isinstance(result, DataSet):
            try:
                tape.record_ingestion(func, args, kwargs, dataset=result)
            except Exception as err:
                raise err
        else:
            # TODO: Do we automatically wrap the returned value into some generic Result and then store it in record_func?
            raise ValueError(f'Tried recording function "{func.__name__}", expected result to be a Result or Dataset but received the {type(result)}')
        
        # Attach the uid of the result to the result
        result._recording_uid = uid

        # Finally return the result
        return result
    record_func._recorded_func = func
    return record_func