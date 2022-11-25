import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
import cr.testing.metric as testing
from cr.data import DataSet, Segmentation
from cr.data.segmentation import ByGroup, ByBins, Temporal, TemporalByBins
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

# ## Create segments###

segmentation_rating = dataset.segment(by='Rating class', method=ByGroup())
segmentation_date = dataset.segment(by='Dato', method=Temporal("yearly"))
segmentation_ep = dataset.segment(by='Erhverv/Privat', method=ByGroup())


# ### Choose what segment to be the inner segment, and the rest outer segment ###

segmentation_inner = segmentation_rating

segmentations_outer = [
    elem for elem in dataset.segmentations if elem != segmentation_inner]

segmentation_cross = dataset.composite_segmentations(*segmentations_outer)

segmentation_exp_p = dataset.segment(
    by='Exposure', method=ByBins([100000, 250000, 1000000]))
segmentation_exp_2 = dataset.segment(
    by='Exposure', method=ByBins(bins=3, method='distance'))
segmentation_exp_3 = dataset.segment(
    by='Exposure', method=ByBins(bins=3, method='observations'))

# ### using the new TemporalByBins
segmentation_date_obs = dataset.segment(
    by='Dato', method=TemporalByBins(bins=3, method='observations'))

# # distance method
segmentation_date_dist = dataset.segment(
    by='Dato', method=TemporalByBins(bins=3, method='distance'))

# #custom method
segmentation_date_custom = dataset.segment(
    by='Dato', method=TemporalByBins(bins=['2019-10-30', '2020-09-29'], time_unit='D'))


