import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
import cr.testing.metric as testing
from cr.testing.result import ResultTable
import cr.calculation as calculate
from cr.data import DataSet, Segmentation
from cr.data.segmentation import ByGroup, Temporal
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

# ### Create Dataset #################################################
dataset = DataSet('data', df_in)

# To do PSI analysis we need to segment the dataset into subset #

segmentation_yearly = dataset.segment(by='Dato', method=Temporal("yearly"))

# Here we make year 2020 our benchmark year (only data up to July in 2021)
segment_benchmark = segmentation_yearly[2020]
# a list of the other segments beside the benchmark
segments = [elem for elem in segmentation_yearly.segments if elem != segment_benchmark]

# calculate psi for factor with few unique variables (categorical variable) ##
col = 'LivingStatus'
print('LivingStatus' in categorical)
psi_c = testing.psi_categorical(
    a=segment_benchmark[col], b=segmentation_yearly[2019][col])
# The psi Result-object has some interesting outputs such as
print("psi value:", psi_c["value"].formatted_value)
print("psi buckets", psi_c["buckets"].formatted_value)
print("psi summands", [round(elem, 4) for elem in psi_c["psi_summands"].value], sum(psi_c["psi_summands"]))
print("color of test is", psi_c["color"].formatted_value, 'since', psi_c["reasoning"].formatted_value)
histogram_psi_c = psi_c["histogram"]
# histogram_psi_c.show()

# calculate psi for factor with many discrete variable (numerical variable) ##
col = 'Age'
print('Age' in numerical)
psi_n = testing.psi_numerical(
    a=segment_benchmark[col], b=segmentation_yearly[2019][col], psi_args={'buckets': 5})
# The psi Result-object has some interesting outputs such as
print("psi value:", psi_n["value"].formatted_value)
print("psi bin edges", psi_n["bin_edges"].formatted_value)
print("psi summands", [round(elem, 4) for elem in psi_n["psi_summands"].value], sum(psi_n["psi_summands"]))
# does not sum since we have some in nan (see histogram plot)
# TODO: should it be possible to get the non_finite information?
print("color of test is", psi_n["color"].formatted_value, 'since', psi_n["reasoning"].formatted_value)
histogram_psi_n = psi_n["histogram"]
# histogram_psi_n.show()

# calculate psi for all factors based on the segments ##
result_table = testing.psi_result_table(
    benchmark_segment=segment_benchmark,
    segments=segments,
    variables=(categorical+numerical),
    variables_categorical=[True]*len(categorical) + [False]*len(numerical),
    column_names=None,
    amber=0.1,
    red=0.25,
    psi_args={'buckets': 5})

df_values = result_table.to_dataframe("value", True)
df_color = result_table.to_dataframe("color", True)

histogram_psi_rt = result_table[f"histogram {result_table.row_names[0]} {result_table.column_names[0]}"]
# histogram_psi_rt.show()

aggregated = testing.aggregate_rag_result_table(result_table)
df_aggregated = aggregated.to_dataframe("value", True)


# ### calculate psi for all factors based without a benchmark segment ##
result_table_no_benchmark = testing.psi_result_table(
    benchmark_segment=None,
    segments=segments,
    variables=(categorical+numerical),
    variables_categorical=[True]*len(categorical) + [False]*len(numerical),
    column_names=None,
    amber=0.1,
    red=0.25,
    psi_args={'buckets': 5})

df_values_no_benchmark = result_table_no_benchmark.to_dataframe("value", True)
df_color_no_benchmark = result_table_no_benchmark.to_dataframe("color", True)

aggregated_no_benchmark = testing.aggregate_rag_result_table(result_table_no_benchmark)
df_aggregated_no_benchmark = aggregated_no_benchmark.to_dataframe("value", True)

