import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
import cr.testing.metric as testing
from cr.data import DataSet, Segmentation
from cr.data.segmentation import ByGroup
import pandas as pd
import numpy as np

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

# ### Choose what factors to used in the concentration ###
name_overwrites = 'D12'

# Create a Result table without inner segment #
rt_all_total = testing.overwrites_result_table(
    dataset=dataset,
    column_name_overwrites=name_overwrites,
    segmentation=None,

)

# ## Get DataFrames for the Result Table ##
df_all_total = rt_all_total.to_dataframe(attribute="value", value=True)
