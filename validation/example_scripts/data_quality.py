import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
import pandas as pd
import numpy as np
from functools import partial
from cr.plotting.plotly import ColorTheme, Theme
from cr.plotting.plotly import (
    figure_histogram,
    figure_bar_histogram,
    unique_values_as_bins,
    add_outlier_box_to_bar_fig
)
from cr.data import DataSet, Segmentation, SourcedArray
from cr.data.segmentation import ByGroup, Temporal

import cr.testing.metric as test
from cr.testing.result import ResultTable

# #####  Plot with plotly default template ##########################################
pio.renderers.default = "browser"
theme = Theme(ColorTheme()).set_as_default()


# ##### IMPORT DATA ###########################################################
name_first = 'Jonas'
name_last = 'Christensen'
df_in = pd.read_parquet(f"C:\\Users\\{name_first}{name_last}\\CR Consulting\\"
                        f"CR Consulting - General\\Tool\\presentation.parquet")
str_cols = df_in.select_dtypes(include=['object']).columns
df_in[str_cols] = df_in[str_cols].applymap(lambda x: np.unicode_(x))


# ### Create Dataset ################################################################
dataset = DataSet('data', df_in)

# Get expected type based on level of measurement: nominal, ordinal, interval or ratio #
# temporary function.


def _propose_factor_types(df):
    from pandas.api import types
    temporals = list(df.select_dtypes(include=['datetime64']).columns)
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Determine initial nominals as non-numeric colums
    nominals = [col for col in df.columns if col not in numeric_cols and col not in temporals]
    ordinals = []
    intervals = []
    ratios = []
    for col in numeric_cols:
        # If only two unique values assume dummy of sort -> nominal
        if len(df[col].unique()) == 2:
            nominals.append(col)
            continue

        # Determine ratios as non-ints
        if not types.is_integer_dtype(df[col].dtype):
            ratios.append(col)
            continue

        # Determine ordinals as sequential ints with no gaps and 'few' unique values
        if df[col].nunique() < 25:
            if np.unique(np.diff(np.sort(df[col].unique()))).size == 1:
                ordinals.append(col)
                continue

        # Otherwise assume it is interval
        intervals.append(col)

    return nominals, ordinals, intervals, ratios, temporals


nominals, ordinals, intervals, ratios, temporals = _propose_factor_types(df_in)
categorical = sorted(nominals + ordinals)
numerical = sorted(intervals + ratios)

# ### Create a Data Quality ResultTable for columns of type interval and/or ratio ####
data_quality = test.data_quality_result_table(
    vectors=dataset[numerical],
    factor_names=numerical,
    missing_amber=0.1,
    missing_red=0.15,
)
df_data_quality_numerical_values = data_quality.to_dataframe("value", True)
df_data_quality_numerical_colors = data_quality.to_dataframe("color", True)

aggregated_numerical = test.aggregate_rag_result_table(data_quality)
df_aggregated_numerical = aggregated_numerical.to_dataframe("value", True)

"""
# Idea: create aggregated table from the function who creates the table
aggregated_numerical = test.data_quality_result_table(
    vectors=dataset[numerical],
    factor_names=numerical,
    missing_amber=0.1,
    missing_red=0.15,
    aggregated=True
)
df_aggregated_numerical = aggregated_numerical.to_dataframe("value", True)
"""

# ### Create a Data Quality ResultTable for columns of type ordinal ####
data_quality_ordinal = test.data_quality_result_table_ordinal(
    vectors=dataset[ordinals],
    factor_names=ordinals,
    missing_amber=0.1,
    missing_red=0.15,
)
df_data_quality_ordinal_values = data_quality_ordinal.to_dataframe("value", True)
df_data_quality_ordinal_colors = data_quality_ordinal.to_dataframe("color", True)

aggregated_ordinal = test.aggregate_rag_result_table(data_quality_ordinal)
df_aggregated_ordinal = aggregated_ordinal.to_dataframe("value", True)


# ### Create a Data Quality ResultTable for columns of type nominal ####
data_quality_nominal = test.data_quality_result_table_nominal(
    vectors=dataset[nominals],
    factor_names=nominals,
    missing_amber=0.1,
    missing_red=0.15,
)
df_data_quality_nominal_values = data_quality_nominal.to_dataframe("value", True)
df_data_quality_nominal_colors = data_quality_nominal.to_dataframe("color", True)

aggregated_nominal = test.aggregate_rag_result_table(data_quality_nominal)
df_aggregated_nominal = aggregated_nominal.to_dataframe("value", True)
