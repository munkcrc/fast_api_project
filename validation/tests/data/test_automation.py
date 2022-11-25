import yaml
import numpy as np
from test_dataset import df, dataset

from cr.automation import recordable, record, get_session_tape, Runner, set_active_tape, Tape
from cr.testing.result import Result, MockResult
from cr.data import DataSet
from cr.data.segmentation import ByGroup

@recordable
def some_function(x, y):
    return Result().add_outputs({"value": x + y})

@recordable
def some_other_function(x:Result, y):
    return Result().add_outputs({"value": x['value'] + y})

@recordable
def some_np_function(x, y, z=1):
    return Result().add_outputs({"value": (x[0] + y[0])*z})

def test_recording_simple():
    with record():
        some_function(3, 4)
        some_function(y=3, x=4, recording_uuid="test")
    
    tape = get_session_tape()
    # Assert that two function have been taped
    keys = list(tape.tests.keys())
    assert len(keys) == 2

    # Check call with args
    assert len(tape.tests[keys[0]]['args']) == 2
    assert 'kwargs' not in tape.tests[keys[0]]
    assert tape.tests[keys[0]]['args'] == [3, 4]

    # Check call with kwargs
    assert len(tape.tests[keys[1]]['kwargs']) == 2
    assert 'args' not in tape.tests[keys[1]]
    assert tape.tests[keys[1]]['kwargs'] == {"y": 3, "x": 4}

    # Check that a key has id test
    assert 'test' in tape.tests
   
def test_result_as_input():
    with record():
        x = some_function(4, 5)
        y = some_other_function(x, 3, recording_uuid="test1")

    serialized_tape = get_session_tape().to_yaml()
    runner = Runner(yaml.safe_load(serialized_tape))
    assert runner.run("test1")["value"] == 12

def test_rerun_numpy(df, dataset):
    # Create the tape
    with record():
        some_np_function(dataset["factor 1"], dataset["factor 2"], recording_uuid="test1")
        some_np_function(dataset["segmentor 2"], dataset["target 2"], 2, recording_uuid="test2")
    tape = get_session_tape()
    serialized_tape = tape.to_yaml()

    # Reload the tape and run
    runner = Runner(yaml.safe_load(serialized_tape), {"dataset": dataset})

    assert runner.run("test1")["value"] == (df["factor 1"][0] + df["factor 2"][0])
    assert runner.run("test2")["value"] == (df["segmentor 2"][0] + df["target 2"][0]) * 2
    
def test_retaping(df, dataset):
    # Create the tape
    with record():
        some_np_function(dataset["factor 1"], dataset["factor 2"], recording_uuid="test1")
        some_np_function(dataset["segmentor 2"], dataset["target 2"], 2, recording_uuid="test2")
    tape = get_session_tape()
    serialized_tape = tape.to_yaml()

    new_tape = Tape()
    set_active_tape(new_tape)
    runner = Runner(yaml.safe_load(serialized_tape), {"dataset": dataset})
    assert runner.run("test1")["value"] == (df["factor 1"][0] + df["factor 2"][0])
    assert len(new_tape.tests["test1"]["args"]) == 2

    assert len(new_tape.tests) == 1
    assert 'test1' in new_tape.tests.keys()

def test_rerun_segmentation(df, dataset):
    # Create the tape
    with record():
        segments = dataset.segment(by="segmentor 1", method=ByGroup())
        for segment in segments:
            if segment.segment_id == "PRIVAT":
                some_np_function(segment["factor 1"], segment["target 1"], recording_uuid="test")

    tape = get_session_tape()
    serialized_tape = tape.to_yaml()

    # Initialize a new dataset
    dataset = DataSet("dataset", df)

    # Reload the tape and run
    runner = Runner(yaml.safe_load(serialized_tape), {"dataset": dataset})

    assert runner.run("test")["value"] == df["factor 1"][2] + df["target 1"][2]

   
def test_dryrun_result_as_input():
    with record():
        x = some_function(4, 5)
        y = some_other_function(x, 3, recording_uuid="test1")

    serialized_tape = get_session_tape().to_yaml()
    runner = Runner(yaml.safe_load(serialized_tape), {"dataset": dataset})
    with record():
        assert type(runner.run("test1", dry_run=True)) == MockResult
