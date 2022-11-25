from typing import Iterable, Optional
import cr.calculation as calculate
from cr.automation import recordable
from cr.testing.result import ScalarRAGResult, ResultTable, ScalarResult, Result
from cr.data import Segment, Segmentation
from cr.testing.output import _format_scalar
from cr.documentation import doc
import cr.plotting.plotly.psi_plot as psi_plot

# from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd


@doc("""The PSI test gives a quantitative measure of how much the data distribution is 
changing between two datasets. Such two datasets could be the historic data used for 
developing a model, and the most recent data the model is used for. """)
@recordable
def psi_numerical(a, b, amber=0.1, red=0.25, psi_args=None, title=None, name_a='a', name_b='b'):
    if len(a) == 0 or len(b) == 0 or np.all(~np.isfinite(a)) or np.all(~np.isfinite(b)):
        psi, dict_intermediate = (np.nan, np.nan)
    else:
        if not psi_args:
            psi_args = {}
        psi, dict_intermediate = calculate.psi_numerical(a, b, **psi_args)

    if psi is None or np.isnan(psi):
        return ScalarResult("PSI", np.nan)

    non_finite = dict_intermediate['non_finite']

    out = ScalarRAGResult("PSI", psi, amber, red).add_outputs({
        "bin_edges": dict_intermediate['bin_edges'],
        "relative_frequency_a": dict_intermediate['relative_frequency']['a'],
        "relative_frequency_b": dict_intermediate['relative_frequency']['b'],
        "psi_summands": dict_intermediate['psi_summands']
    })

    def get_hist_fig(output, dict_non_finite):
        psi_value = output["value"].value
        bin_edges = output["bin_edges"].value
        relative_frequency_a = list(output["relative_frequency_a"].value)
        relative_frequency_b = list(output["relative_frequency_b"].value)
        psi_summands = list(output["psi_summands"].value)

        buckets = psi_plot.get_buckets(bin_edges)
        fig = psi_plot.figure_psi_deep_dive(
            psi_value, buckets,
            relative_frequency_a, relative_frequency_b,
            psi_summands, dict_non_finite,
            title=title, name_a=name_a, name_b=name_b)

        return fig

    return out.add_outputs({
        "histogram": lambda output=out: get_hist_fig(output, non_finite)
    })


@doc("""The PSI test gives a quantitative measure of how much the data distribution is 
changing between two datasets. Such two datasets could be the historic data used for 
developing a model, and the most recent data the model is used for. """)
@recordable
def psi_categorical(a, b, amber=0.1, red=0.25, title=None, name_a='a', name_b='b'):
    if len(a) == 0 or len(b) == 0:
        psi, dict_intermediate = (np.nan, np.nan)
    elif np.issubdtype(a.dtype, np.number) and np.all(~np.isfinite(a)):
        psi, dict_intermediate = (np.nan, np.nan)
    elif not np.issubdtype(a.dtype, np.number) and np.all(
            pd.isna(a) | np.isin(a, ('nan', 'None', '', '<NA>'))):
        psi, dict_intermediate = (np.nan, np.nan)
    elif np.issubdtype(b.dtype, np.number) and np.all(~np.isfinite(b)):
        psi, dict_intermediate = (np.nan, np.nan)
    elif not np.issubdtype(b.dtype, np.number) and np.all(
            pd.isna(b) | np.isin(b, ('nan', 'None', '', '<NA>'))):
        psi, dict_intermediate = (np.nan, np.nan)
    else:
        psi, dict_intermediate = calculate.psi_categorical(a, b)

    if psi is None or np.isnan(psi):
        return ScalarResult("PSI", np.nan)

    out = ScalarRAGResult("PSI", psi, amber, red).add_outputs({
        "buckets": dict_intermediate['buckets'],
        "relative_frequency_a": dict_intermediate['relative_frequency']['a'],
        "relative_frequency_b": dict_intermediate['relative_frequency']['b'],
        "psi_summands": dict_intermediate['psi_summands']
    })

    def get_hist_fig(output):
        psi_value = output["value"].value
        buckets = output["buckets"].value
        relative_frequency_a = output["relative_frequency_a"]
        relative_frequency_b = output["relative_frequency_b"]
        psi_summands = output["psi_summands"]

        fig = psi_plot.figure_psi_deep_dive(
            psi_value, buckets,
            relative_frequency_a, relative_frequency_b,
            psi_summands,
            title=title, name_a=name_a, name_b=name_b)

        return fig

    return out.add_outputs({
        "histogram": lambda output=out: get_hist_fig(output)
    })


@recordable
def psi_result_table(
        benchmark_segment: Optional[Segment],
        segments: Iterable[Segment],
        variables: Iterable[str],
        variables_categorical: Optional[Iterable[bool]] = None,
        benchmark_segment_name: Optional[str] = None,
        segment_names: Optional[Iterable[str]] = None,
        amber=0.1,
        red=0.25,
        psi_args=None) -> ResultTable:

    if not psi_args:
        psi_args = {}

    if variables_categorical:
        categorical_dict = {
            variable: categorical for
            variable, categorical in zip(variables, variables_categorical)}
    else:
        categorical_dict = {variable: None for variable in variables}

    def _psi(segment_a, segment_b, categorical, variable, name_a, name_b):
        vector_a = segment_a[variable]
        vector_b = segment_b[variable]
        categorical = categorical[variable]
        title = (f"PSI deep dive for {variable} on subsets "
                 f"{segment_a.segment_id} vs. {segment_b.segment_id}")
        if categorical is None:
            if len(np.unique(vector_a)) < 11 and len(np.unique(vector_b)) < 11:
                categorical = True
            else:
                categorical = False

        if categorical:
            return psi_categorical(
                vector_a, vector_b, amber=amber, red=red,
                title=title, name_a=name_a, name_b=name_b)
        else:
            return psi_numerical(
                vector_a, vector_b, amber=amber, red=red, psi_args=psi_args,
                title=title, name_a=name_a, name_b=name_b)

    if benchmark_segment is None:
        if segment_names is None:
            column_names = [
                f"{segment_a.segment_id} vs. {segment_b.segment_id}"
                for segment_a, segment_b in zip(segments[:-1], segments[1:])]
            results_array = np.array([
                [_psi(segment_a, segment_b, categorical_dict, variable,
                      f"{segment_a.segment_id}", f"{segment_b.segment_id}")
                 for segment_a, segment_b in zip(segments[:-1], segments[1:])]
                for variable in variables], dtype=object)
        else:
            column_names = [
                f"{column_a} vs. {column_b}"
                for column_a, column_b in zip(segment_names[:-1], segment_names[1:])]
            results_array = np.array([
                [_psi(segment_a, segment_b, categorical_dict, variable,
                      column_a, column_b)
                 for segment_a, segment_b, column_a, column_b in
                 zip(segments[:-1], segments[1:], segment_names[:-1],
                     segment_names[1:])]
                for variable in variables], dtype=object)

    else:
        if segment_names is not None and benchmark_segment_name is not None:
            column_names = segment_names
            results_array = np.array([
                [_psi(benchmark_segment, segment, categorical_dict, variable,
                      benchmark_segment_name, column_b) for segment, column_b in
                 zip(segments, segment_names)]
                for variable in variables], dtype=object)

        else:
            column_names = [segment.segment_id for segment in segments]
            results_array = np.array([
                [_psi(benchmark_segment, segment, categorical_dict, variable,
                      f"{benchmark_segment.segment_id}", f"{segment.segment_id}") for
                 segment in segments]
                for variable in variables], dtype=object)

    column_names = [tuple(elem) if isinstance(elem, list) else elem
                    for elem in column_names]

    rt = ResultTable(
        name='PSI',
        row_names=list(variables),
        column_names=column_names,
        results=results_array
    )

    def histogram(rt_in, row_in, column_in):
        row_index = rt_in.row_names.index(row_in)
        column_index = rt_in.column_names.index(column_in)
        psi_temp = rt_in.results.value[row_index][column_index]
        fig_out = psi_temp["histogram"]
        return fig_out

    histogram_dict = {
        f'histogram {row} {column}':
            lambda rt_in=rt, row_in=row, column_in=column: histogram(rt_in, row_in, column_in)
        for row in rt.row_names for column in rt.column_names}
    rt.add_outputs(histogram_dict)
    return rt

