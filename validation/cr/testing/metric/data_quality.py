from functools import partial
from typing import Callable, Iterable, Union

import numpy as np

import cr.calculation as calculate
from cr.automation import recordable
from cr.documentation import doc
from cr.plotting.plotly import data_quality_plots as dqp
import cr.testing.metric.simple as simple
from cr.testing.result import ResultTable, ScalarRAGResult, ScalarResult


@doc("""Asymmetric Tukey's fence is an outlier method based on deviation from the mean.
It determines outliers as anything outside the region [Q1-2*k*(Q2-Q1), Q3+2*k*(Q3-Q2)].
here k is a constant that can be tuned to control the sensitivity of the method, common choices are 1.5 and 3. 
""",
     output_docs={
         "value": "Number of outliers in total",
         "nr_below": "Number of outliers below the lower bound",
         "nr_above": "Number of outliers above the upper bound",
         "lower_bound": "The lower bound, anything below is an outlier",
         "upper_bound": "The upper bound, anything above is an outlier",
         "constant": "The constant k"
     })
@recordable
def outliers_asymmetric_tukey_fences(v, k=3, amber=0.1, red=0.15):
    # TODO: ensure that factors with few discrete values do not get any outliers.
    #  i.e. for a factor with only two variables (0,1) it could happened that all the
    #  1's could be seen as outliers. This should just be avoided all together.
    nr_of_outliers, nr_below, nr_above, lower, upper = calculate.outlier_asymmetric_tukey_fences(v, k)
    if nr_of_outliers is None or np.isnan(nr_of_outliers):
        return ScalarResult("OUTLIERS", np.nan)
    return ScalarRAGResult(
        "OUTLIERS", nr_of_outliers, v.size * amber, v.size * red).add_outputs(
            {"nr_below": nr_below,
             "nr_above": nr_above,
             "lower_bound": lower,
             "upper_bound": upper,
             "constant": k})


@doc("""Tukey's fence is an outlier method based on the interquantile range.
It determines outliers as anything outside the region [Q1-k*IQR, Q3+k*IQR].
here k is a constant that can be tuned to control the sensitivity of the method, common choices are 1.5 and 3. 
""",
     output_docs={
         "value": "Number of outliers in total",
         "nr_below": "Number of outliers below the lower bound",
         "nr_above": "Number of outliers above the upper bound",
         "lower_bound": "The lower bound, anything below is an outlier",
         "upper_bound": "The upper bound, anything above is an outlier",
         "constant": "The constant k"
     })
@recordable
def outliers_tukey_fences(v, k=3, amber=0.1, red=0.15):
    # TODO: ensure that factors with few discrete values do not get any outliers.
    #  i.e. for a factor with only two variables (0,1) it could happened that all the
    #  1's could be seen as outliers. This should just be avoided all together.
    nr_of_outliers, nr_below, nr_above, lower, upper = calculate.outlier_tukey_fences(v, k)
    if nr_of_outliers is None or np.isnan(nr_of_outliers):
        return ScalarResult("OUTLIERS", np.nan)
    return ScalarRAGResult(
        "OUTLIERS", nr_of_outliers, v.size * amber, v.size * red).add_outputs(
            {"nr_below": nr_below,
             "nr_above": nr_above,
             "lower_bound": lower,
             "upper_bound": upper,
             "constant": k})


@recordable
def data_quality_result_table(
        vectors: Union[Iterable[np.ndarray], np.ndarray],
        factor_names: Iterable[str],
        outlier_function: Callable = outliers_tukey_fences,
        missing_amber=0.1,
        missing_red=0.15,

) -> ResultTable:
    """
            The ResultTable showing Data Quality
            vectors: the i'th vector in vectors represent the values for the i'th factor
                and is used in the i'th row as input
            factor_names: the i'th factor_name in factor_names represent the name for
                the i'th factor
            metric_functions: the j'th metric_function in metric_functions is used to
                calculate the results in the j'th column.

            The structure is a follows:

                for notation denote:
                result[i][j] = metric_functions[j](vectors[i])
                result_name[j] = metric_functions[j](vectors[i]).name.value


                             result_name[0], result_name[1], ..., result_name[k]
            factor_names[0]   results[0][0],  results[0][1], ...,  results[0][k]
            factor_names[1]   results[1][0],  results[1][1], ...,  results[1][k]
            ...
            factor_names[n]   results[n][0],   results[n][1], ...,   results[n][k]

        """
    metric_functions = [
        partial(simple.missing, amber=missing_amber, red=missing_red),
        simple.unique_values,
        simple.minimum_value,
        partial(simple.percentile, q=10),
        simple.median,
        partial(simple.percentile, q=90),
        simple.maximum_value,
        simple.mean_value,
        outlier_function
    ]
    results = np.array(
        [[metric_function(vector) for metric_function in metric_functions]
         for vector in vectors]
    )
    result_names = [result.name.value for result in results[0]]

    rt_out = ResultTable(
        'DATA QUALITY', list(factor_names), result_names, results).add_outputs(
        {'vectors': vectors})

    def histogram(rt, name):
        index_factor = rt.row_names.index(name)
        index_unique = rt.column_names.index('UNIQUE')
        v = rt['vectors'].value[index_factor]
        u = rt['results'].value[index_factor][index_unique]["values"].value
        c = rt['results'].value[index_factor][index_unique]["counts"].value

        hist_type = dqp.unique_values_as_bins(v, unique_values=u,
                                              unique_values_count=c)
        fig_out = dqp.figure_bar_histogram(
            v=v, name=name, norm='probability',
            unique_value_bins=hist_type,
            unique_values=u, unique_values_count=c)

        if len(u) > 20:
            index_outliers = rt.column_names.index('OUTLIERS')
            lb = rt['results'].value[index_factor][index_outliers]["lower_bound"]  # .value
            ub = rt['results'].value[index_factor][index_outliers]["upper_bound"]  # .value
            dqp.add_outlier_box_to_bar_fig(fig_out, lb, ub)

        return fig_out

    def histogram_wo_outliers(rt, name):
        index_factor = rt.row_names.index(name)
        v = rt['vectors'].value[index_factor]
        v_no_nan = v[~np.isnan(v)]

        index_outliers = rt.column_names.index('OUTLIERS')
        # TODO: make it possible for output Output(object) to compare with value.
        #  v_no_nan <= ub and v_no_nan >= lb does not work correct (returns one bool)
        #  v_no_nan.__le__(ub) , v_no_nan.__ge__(lb) (it might actually have to do with
        #  np.ndarray __le__ when it gets a Output() hmm.
        #  https: // numpy.org / devdocs / user / basics.dispatch.html
        lb = rt['results'].value[index_factor][index_outliers]["lower_bound"].value
        ub = rt['results'].value[index_factor][index_outliers]["upper_bound"].value
        v_wo_outliers = v_no_nan[(lb <= v_no_nan) & (v_no_nan <= ub)]

        u, c = np.unique(v_wo_outliers, return_counts=True)

        hist_type = dqp.unique_values_as_bins(v_wo_outliers, unique_values=u,
                                              unique_values_count=c)
        # TODO: change structure of functions to do histogram plot so we get correct
        #  title for a histogram plot without outliers
        fig_out = dqp.figure_bar_histogram(
            v=v_wo_outliers, name=name, norm='probability',
            unique_value_bins=hist_type,
            unique_values=u, unique_values_count=c)
        dqp.add_sub_title(fig_out, "Where outliers have been excluded")
        return fig_out

    histogram_dict = {
        f'histogram {factor_name}': lambda rt=rt_out, name=factor_name: histogram(rt, name)
        for i, factor_name in enumerate(rt_out.row_names)}
    rt_out.add_outputs(histogram_dict)

    histogram_wo_outliers_dict = {
        f'histogram {factor_name} wo outliers':
            lambda rt=rt_out, name=factor_name: histogram_wo_outliers(rt, name)
        for i, factor_name in enumerate(rt_out.row_names)}
    rt_out.add_outputs(histogram_wo_outliers_dict)
    rt_out._metric_functions = metric_functions

    return rt_out


@recordable
def data_quality_result_table_nominal(
        vectors: Union[Iterable[np.ndarray], np.ndarray],
        factor_names: Iterable[str],
        missing_amber=0.1,
        missing_red=0.15,
) -> ResultTable:
    """
            The ResultTable showing Data Quality
            vectors: the i'th vector in vectors represent the values for the i'th factor
                and is used in the i'th row as input
            factor_names: the i'th factor_name in factor_names represent the name for
                the i'th factor
            metric_functions: the j'th metric_function in metric_functions is used to
                calculate the results in the j'th column.

            The structure is a follows:

                for notation denote:
                result[i][j] = metric_functions[j](vectors[i])
                result_name[j] = metric_functions[j](vectors[i]).name.value


                             result_name[0], result_name[1], ..., result_name[k]
            factor_names[0]   results[0][0],  results[0][1], ...,  results[0][k]
            factor_names[1]   results[1][0],  results[1][1], ...,  results[1][k]
            ...
            factor_names[n]   results[n][0],   results[n][1], ...,   results[n][k]

        """
    metric_functions = [
        partial(simple.missing, amber=missing_amber, red=missing_red),
        simple.unique_values,
        simple.mode
    ]
    results = np.array(
        [[metric_function(vector) for metric_function in metric_functions]
         for vector in vectors]
    )
    result_names = [result.name.value for result in results[0]]

    rt_out = ResultTable(
        'NOMINAL DATA QUALITY', list(factor_names), result_names, results).add_outputs(
        {'vectors': vectors})

    def histogram(rt, name):
        index_factor = rt.row_names.index(name)
        index_unique = rt.column_names.index('UNIQUE')
        v = rt['vectors'].value[index_factor]
        u = rt['results'].value[index_factor][index_unique]["values"].value
        c = rt['results'].value[index_factor][index_unique]["counts"].value

        fig_out = dqp.figure_bar_histogram(
            v=v, name=name, norm='count',
            unique_value_bins=True,
            unique_values=u, unique_values_count=c)
        fig_out.update_layout(xaxis={'type': 'category'})

        return fig_out

    histogram_dict = {
        f'histogram {factor_name}': lambda rt=rt_out, name=factor_name: histogram(rt, name)
        for i, factor_name in enumerate(rt_out.row_names)}
    rt_out.add_outputs(histogram_dict)
    rt_out._metric_functions = metric_functions

    return rt_out


@recordable
def data_quality_result_table_ordinal(
        vectors: Union[Iterable[np.ndarray], np.ndarray],
        factor_names: Iterable[str],
        missing_amber=0.1,
        missing_red=0.15,
) -> ResultTable:
    """
            The ResultTable showing Data Quality
            vectors: the i'th vector in vectors represent the values for the i'th factor
                and is used in the i'th row as input
            factor_names: the i'th factor_name in factor_names represent the name for
                the i'th factor
            metric_functions: the j'th metric_function in metric_functions is used to
                calculate the results in the j'th column.

            The structure is a follows:

                for notation denote:
                result[i][j] = metric_functions[j](vectors[i])
                result_name[j] = metric_functions[j](vectors[i]).name.value


                             result_name[0], result_name[1], ..., result_name[k]
            factor_names[0]   results[0][0],  results[0][1], ...,  results[0][k]
            factor_names[1]   results[1][0],  results[1][1], ...,  results[1][k]
            ...
            factor_names[n]   results[n][0],   results[n][1], ...,   results[n][k]

        """
    metric_functions = [
        partial(simple.missing, amber=missing_amber, red=missing_red),
        simple.unique_values,
        simple.mode,
        simple.median
    ]
    results = np.array(
        [[metric_function(vector) for metric_function in metric_functions]
         for vector in vectors]
    )
    result_names = [result.name.value for result in results[0]]

    rt_out = ResultTable(
        'ORDINAL DATA QUALITY', list(factor_names), result_names, results).add_outputs(
        {'vectors': vectors})

    def histogram(rt, name):
        index_factor = rt.row_names.index(name)
        index_unique = rt.column_names.index('UNIQUE')
        v = rt['vectors'].value[index_factor]
        u = rt['results'].value[index_factor][index_unique]["values"].value
        c = rt['results'].value[index_factor][index_unique]["counts"].value

        fig_out = dqp.figure_bar_histogram(
            v=v, name=name, norm='count',
            unique_value_bins=True,
            unique_values=u, unique_values_count=c)
        fig_out.update_layout(xaxis={'type': 'category'})

        return fig_out

    histogram_dict = {
        f'histogram {factor_name}': lambda rt=rt_out, name=factor_name: histogram(rt, name)
        for i, factor_name in enumerate(rt_out.row_names)}
    rt_out.add_outputs(histogram_dict)
    rt_out._metric_functions = metric_functions

    return rt_out


@recordable
def aggregate_rag_result_table(result_table, split_on_column=True):
    import pandas as pd
    from cr.testing.result import RAGResult
    result_table_input = result_table
    df_colors = result_table_input.to_dataframe("color", True)
    df_values = result_table_input.to_dataframe("value", True)

    row_names = ["GREEN", "AMBER", "RED", "nan", "TOTAL"]
    results = []
    column_names = []
    # col = df_colors.columns[0]
    if split_on_column:
        for col in df_colors:
            array = df_colors[col].values
            mask_value_not_nan = pd.notna(df_values[col])
            metrics = result_table_input.get_result_subset(column_names=col)
            # Convert from 2-d numpy array to list:
            metrics = list(np.squeeze(metrics))
            mask_rag_result = [isinstance(elem, RAGResult) for elem in metrics]
            if not np.all(~mask_value_not_nan) and np.any(mask_rag_result):
                results.append(
                    [simple.count(array[(array == row_names[0]) & mask_value_not_nan]),
                     simple.count(array[(array == row_names[1]) & mask_value_not_nan]),
                     simple.count(array[(array == row_names[2]) & mask_value_not_nan]),
                     simple.count(array[~mask_value_not_nan]),
                     simple.count(array)])
                column_names.append(col)
    else:
        array = df_colors.values.flatten('F')
        mask_value_not_nan = pd.notna(df_values.values.flatten('F'))
        metrics = result_table_input.results.value.flatten('F')
        mask_rag_result = [isinstance(elem, RAGResult) for elem in metrics]
        if not np.all(~mask_value_not_nan) and np.any(mask_rag_result):
            results.append(
                [simple.count(array[(array == row_names[0]) & mask_value_not_nan]),
                 simple.count(array[(array == row_names[1]) & mask_value_not_nan]),
                 simple.count(array[(array == row_names[2]) & mask_value_not_nan]),
                 simple.count(array[~mask_value_not_nan]),
                 simple.count(array)])
            column_names = ['ALL']

    return ResultTable(
        name='AGGREGATE',
        row_names=row_names,
        column_names=column_names,
        results=np.array(results).T)

