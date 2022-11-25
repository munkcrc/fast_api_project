import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
import cr.testing.metric as testing
import cr.calculation as calculate
from cr.data import DataSet, Segmentation
from cr.data.segmentation import ByGroup
import pandas as pd
import numpy as np
from datetime import date, datetime

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
dataset = DataSet('data', df_in)

# Create a ResultTable (using all rows in dataset), representing a migration matrix #
rt_all = testing.migration_matrix(
    start=dataset['FROM'],
    end=dataset['TO'],
    order=None,
    include_all=False
)
# ## Get DataFrames with the migration probability and -counts ##
df_all_prob = rt_all.to_dataframe(attribute="value", value=True)
df_all_count = rt_all.to_dataframe(attribute="count", value=True)

# Create a ResultTable, representing stability tests from the migration matrix
ignore_states = ['D', 'None', 'OUT']
rt_all_stability = testing.matrix_stability(migration=rt_all, ignore_states=ignore_states)
# Each entry in the rt_all_stability.result is a hypothesis object
# ## Get DataFrames with the p-value, color and h0 ##
df_all_stability_pval = rt_all_stability.to_dataframe(attribute="value", value=True)
df_all_stability_color = rt_all_stability.to_dataframe(attribute="color", value=True)
df_all_stability_h0 = rt_all_stability.to_dataframe(attribute="h0", value=True)

# TODO see following:
#  Notice that the stability tests only makes sense to do on non-defaulted rating groups
#  (i.e for rating group 1-10 and not for 'D' and 'OUT). We might need to create some
#  input argument that can ensure only non-defaulted rating groups are used

# Create a ResultTable, representing stability tests for the default column in the
# migration matrix
index_default = rt_all.column_names.index('D')
rt_all_stability_default = testing.matrix_stability_column(migration=rt_all,
                                                           index_column=index_default)
# Each entry in the rt_all_stability_default.result is a hypothesis object
# ## Get DataFrames with the p-value, color and h0 ##
df_all_stability_pval_default = rt_all_stability_default.to_dataframe(attribute="value", value=True)
df_all_stability_color_default = rt_all_stability_default.to_dataframe(attribute="color", value=True)
df_all_stability_h0_default = rt_all_stability_default.to_dataframe(attribute="h0", value=True)

# It is also possible to calculate the matrix weighted bandwidth for the upper and lower
# diagonal for non-defaulted rating classes, but it is not implemented in testing yet:
mwb_migration_input = df_all_count.values[0:10, 0:10]
mwb_upper = calculate.matrix_weighted_bandwidth(migration_count=mwb_migration_input,
                                                upper=True)
mwb_lower = calculate.matrix_weighted_bandwidth(migration_count=mwb_migration_input,
                                                upper=False)

# All of the following can be applied on sub-sets of the Dataset.

segmentation_date = dataset.segment(by='Dato', method=ByGroup())
segmentation_ep = dataset.segment(by='Erhverv/Privat', method=ByGroup())

segmentation_cross = dataset.composite_segmentations(
    *[elem for elem in dataset.segmentations])

# get the migration matrix for Privat in '2019-06-30':
rt_2019_privat = testing.migration_matrix(
    start=segmentation_cross[[np.datetime64(date(2019, 6, 30)), 'Privat']]['FROM'],
    end=segmentation_cross[[np.datetime64('2019-06-30'), 'Privat']]['TO'],
    order=None,
    include_all=False
)
df_2019_privat_prob = rt_2019_privat.to_dataframe(attribute="value", value=True)
df_2019_privat_count = rt_2019_privat.to_dataframe(attribute="count", value=True)

# etc etc.

