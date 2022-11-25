import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
import cr.testing.metric as testing
from cr.data import DataSet, Segmentation
from cr.data.segmentation import ByGroup, ByBins, Temporal
import pandas as pd
import numpy as np
from datetime import date

# #####  Plot with plotly default template #####################################
pio.renderers.default = "browser"
theme = Theme(ColorTheme()).set_as_default()

# ##### IMPORT DATA ###########################################################
name_first = 'Jonathan'
name_last = 'Kofod'
df_in = pd.read_parquet(f"C:\\Users\\{name_first}{name_last}\\CR Consulting\\"
                        f"CR Consulting - General\\Tool\\presentation.parquet")
str_cols = df_in.select_dtypes(include=['object']).columns
df_in[str_cols] = df_in[str_cols].applymap(lambda x: np.unicode_(x))

# ### Create Dataset #################################################
dataset_original = DataSet('data', df_in)

# ### Create segmentation to remove all already defaulted #############################

segmentation_defaulted = dataset_original.segment(by='Defaulted', method=ByGroup())

dataset = segmentation_defaulted[0]

# ### Choose what factors to used in the backtest ###
name_prediction = 'PD 12M'
name_actual = 'D12'
name_exposure = 'Exposure'

# Create a Result table without inner segment #
rt_all_total = testing.pd_back_test_result_table(
    dataset=dataset,
    column_name_exposure=name_exposure,
    column_name_defaulted=name_actual,
    column_name_pd=name_prediction,
    segmentation=None,
    red=0.05,
    amber=0.05 * 2
)
# ## Get DataFrames for the Result Table ##
df_all_total = rt_all_total.to_dataframe(attribute="value", value=True)

# ## Create segments to use for pd back test ###

segmentation_rating = dataset.segment(by='Rating class', method=ByGroup())
segmentation_date = dataset.segment(by='Dato', method=Temporal("yearly"))
segmentation_ep = dataset.segment(by='Erhverv/Privat', method=ByGroup())


# ### Choose what segment to be the inner segment, and the rest outer segment ###

segmentation_inner = segmentation_rating

segmentations_outer = [
    elem for elem in dataset.segmentations if elem != segmentation_inner]

segmentation_cross = dataset.composite_segmentations(*segmentations_outer)


# Create a Result table with inner segment on a specific sub_set#
sub_set = [2019, 'Privat']
temp_d = segmentation_cross[sub_set]
temp_s = temp_d.segment(by=segmentation_inner.by,
                        method=segmentation_inner.method)
rt_2019_privat_inner = testing.pd_back_test_result_table(
    dataset=temp_d,
    column_name_exposure=name_exposure,
    column_name_defaulted=name_actual,
    column_name_pd=name_prediction,
    segmentation=temp_s,
    red=0.05,
    amber=0.05 * 2
)

# ## Get DataFrames for the Result Table ##
df_2019_privat_inner = rt_2019_privat_inner.to_dataframe(attribute="value", value=True)

# Calculate pd back test for all combinations

segmentation_exp_p = dataset.segment(
    by='Exposure', method=ByBins([100000, 250000, 1000000]))

segmentation_inner = segmentation_rating

segmentations_outer = [
    elem for elem in dataset.segmentations if elem != segmentation_inner]

segmentation_cross = dataset.composite_segmentations(*segmentations_outer)

_date_to_str = lambda elem: str(elem) if isinstance(elem, date) else elem

_results = {}
# (lave out cross_segment where obs=0, and where all pd or all dfs are nans
for cross_segment in segmentation_cross:
    if cross_segment.observations > 0 \
            and not np.all(np.isnan(cross_segment[name_prediction])) \
            and not np.all(np.isnan(cross_segment[name_actual])):
        if isinstance(cross_segment.segment_id, list):
            key = tuple([_date_to_str(elem) for elem in cross_segment.segment_id])
        else:
            key = _date_to_str(cross_segment.segment_id)

        rt_temp = testing.pd_back_test_result_table(
            dataset=cross_segment,
            column_name_exposure=name_exposure,
            column_name_defaulted=name_actual,
            column_name_pd=name_prediction,
            segmentation=cross_segment.segment(
                by=segmentation_inner.by,
                method=segmentation_inner.method),
            red=0.05,
            amber=0.05 * 2
        )

        df_temp = rt_temp.to_dataframe("value", value=True)
        # df_temp = df_temp.applymap(lambda x: x.formatted_value)
        # df.reset_index(inplace=True)

        _results[key] = [rt_temp, df_temp]
