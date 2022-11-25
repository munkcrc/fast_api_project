import pytest
import numpy as np
import pandas as pd
import io
from datetime import datetime
from cr.data import DataSet
from cr.data.segmentation import ByGroup, ByBins, Temporal
from cr.data.segmentation.ordinal import get_bins_with_equally_many_observations

def get_df():
    csv_data = """
target 1;target 2;factor 1;factor 2;segmentor 1;segmentor 2
1;0.506050764;6;8.500470157;ERHVERV;2016
1;0.820553433;7;1.856536542;ERHVERV;2011
1;0.871559982;9;3.487774988;PRIVAT;2016
0;0.186515687;8;3.713330483;PRIVAT;2013
1;0.094145296;6;10.08363758;PRIVAT;2019
1;0.316498373;8;8.152383426;ERHVERV;2020
1;0.929669618;4;6.307885408;ERHVERV;2019
0;0.299820652;2;4.030830161;ERHVERV;2020
    """
    df = pd.read_csv(io.StringIO(csv_data), delimiter=";")
    df["segmentor 3"] = df["segmentor 2"].map(lambda x: datetime(x, 1, 1))
    return df

@pytest.fixture
def df():
    return get_df()

@pytest.fixture
def dataset():
    return DataSet("dataset", get_df())

def test_indexing(dataset, df):
    np.testing.assert_array_equal(dataset['target 1'], df["target 1"].values)
    np.testing.assert_array_equal(dataset['factor 1'], df["factor 1"].values)
    np.testing.assert_array_equal(dataset['segmentor 1'], df["segmentor 1"].values)

def test_nominal_segments_distinct(dataset, df):
    segments = dataset.segment(by="segmentor 1", method=ByGroup()).segments
    assert len(segments) == 2
    assert segments[0].by == "segmentor 1"

    if segments[0].segment_id == "ERHVERV":
        erhv_seg = 0
        priv_seg = 1
    else:
        erhv_seg = 1
        priv_seg = 0

    assert np.all(segments[erhv_seg]["segmentor 1"] == "ERHVERV")
    assert np.all(segments[priv_seg]["segmentor 1"] == "PRIVAT")
    np.testing.assert_array_equal(segments[erhv_seg]["factor 1"], df[df["segmentor 1"]=="ERHVERV"]["factor 1"].values)
    np.testing.assert_array_equal(segments[priv_seg]["factor 1"], df[df["segmentor 1"]=="PRIVAT"]["factor 1"].values)

def test_nominal_segments_groups(dataset):
    segments = dataset.segment(by="segmentor 2", method=ByGroup(groups=[[2011, 2013, 2016], [2019, 2020]])).segments
    assert len(segments) == 2
    assert segments[0].by == "segmentor 2"

    if segments[0].segment_id == [2011, 2013, 2016]:
        early_seg = 0
        late_seg = 1
    else:
        early_seg = 1
        late_seg = 0

    assert np.all(np.isin(segments[early_seg]["segmentor 2"], [2011, 2013, 2016]))
    assert np.all(np.isin(segments[late_seg]["segmentor 2"], [2019, 2020]))

def test_ordinal_segments_bins(dataset):
    segments = dataset.segment(by="segmentor 2", method=ByBins(bins=[2017])).segments
    assert len(segments) == 2
    assert segments[0].by == "segmentor 2"

    if segments[0].segment_id == '[-inf, 2017)':
        early_seg = 0
        late_seg = 1
    else:
        early_seg = 1
        late_seg = 0

    assert np.all(np.isin(segments[early_seg]["segmentor 2"], [2011, 2013, 2016]))
    assert np.all(np.isin(segments[late_seg]["segmentor 2"], [2019, 2020]))

def test_ordinal_segments_bins(dataset):
    segments = dataset.segment(by="segmentor 2", method=ByBins(bins=[2017])).segments
    assert len(segments) == 2
    assert segments[0].by == "segmentor 2"

    
def test_temporal_segments(dataset):
    segments = dataset.segment(by="segmentor 3", method=Temporal(frequency="yearly")).segments
    assert len(segments) == 5

def test_composite_segments(dataset):
    segmentation1 = dataset.segment(by="segmentor 3", method=Temporal(frequency="yearly"))
    segmentation2 = dataset.segment(by="segmentor 1", method=ByGroup())
    segments = segmentation1.composite_with(segmentation2).segments

    assert len(segments) == 10 # 5 years + 2 groups


@pytest.mark.parametrize('input_x,input_nr_of_bins,expected', [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8], 3, [2.5, 5.5]),
    ([0, 1, 2, 2, 3, 4, 4, 5, 6], 3, [1.5, 3.5]),
    ([0, 1, 2, 2, 3, 4, 4, 4, 4], 3, [1.5, 3.5]),
    ([0, 0, 0, 0, 0, 0, 3, 5, 5], 3, [1.5, 4.0]),
    ([0, 0, 0, 0, 0, 0, 5, 5, 5], 3, [2.5]),
    ([0, 1, 3, 3, 3, 3, 3, 3, 3], 3, [0.5, 2.0]),
    ([0, 1, 3, 3, 3, 3, 3, 3, 4], 3, [2.0, 3.5]),
    ([0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5], 3, [2.0, 3.5]),
    ([0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5], 4, [0.5, 2.0, 3.5]),
    ([0, 0, 0, 0, 0, 0, 3, 5, 6], 3, [1.5, 4.0]),
    ([0, 1, 3, 6, 6, 6, 6, 6, 6], 3, [0.5, 4.5]),
    ([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 6, 6, 6, 6, 6, 6], 3,
     [0.5, 4.5],),
    ([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 6, 6, 6, 6, 6, 6], 4,
     [0.5, 2.5, 4.5]),
    ([1] * 4 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 5 + [7] * 70 + [8], 3,
     [3.5, 6.5]),
    ([1] * 4 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 5 + [7] * 70 + [8], 4,
     [2.5, 4.5, 6.5]),
    ([1] * 4 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 5 + [7] * 70 + [8], 5,
     [2.5, 3.5, 4.5, 6.5]),
    ([1] * 4 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 0 + [6] * 16 + [7] * 70 + [8], 3,
     [5.0, 6.5]),
    ([1] * 4 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 0 + [6] * 16 + [7] * 70 + [8], 4,
     [2.5, 5.0, 6.5]),
    ([1] * 4 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 0 + [6] * 16 + [7] * 70 + [8], 5,
     [1.5, 2.5, 5.0, 6.5])
])
def test_get_equally_spaced_bins(input_x, input_nr_of_bins, expected):
    np.testing.assert_array_equal(
        get_bins_with_equally_many_observations(input_x, input_nr_of_bins), expected)

