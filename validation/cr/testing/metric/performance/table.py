from typing import Sequence, Callable, Optional

import numpy as np

from cr import data as data
from cr.automation import recordable
import cr.testing.metric as metric

from cr.testing.result import ResultTable, ScalarResult
from functools import partial


# TODO: we can not use recordable since cr.automation.recordable.recordable( ) only
#  works for Result or DataSet
# @recordable
def calc_row_result(data_input, column_names, functions, data_columns, pre_compute):
    column_index_pre = [i for i, pre in enumerate(pre_compute) if pre]
    column_index_post = [i for i, pre in enumerate(pre_compute) if not pre]

    # put each element in data_columns into a list.
    # Needed for __getitem__ in Dataset and Segment
    data_columns = [
        [cols] if isinstance(cols, str) else list(cols) for cols in data_columns]

    # Calculate the pre-functions for either a DataSet/Segment or a Segmentation
    if isinstance(data_input, data.DataSet):
        results_array = np.empty(shape=(len(column_names), 1), dtype=object)
        results_array[column_index_pre, :] = [
            [func(*data_input[cols])] for func, cols, compute in
            zip(functions, data_columns, pre_compute) if compute
        ]
    else:
        results_array = np.empty(
            shape=(len(column_names), len(data_input.segments)), dtype=object)
        results_array[column_index_pre, :] = [
            [func(*subset[cols]) for subset in data_input]
            for func, cols, compute in
            zip(functions, data_columns, pre_compute) if compute]

    # get the column indices needed for each post function
    column_indices = [
        [column_names.index(elem) for elem in cols]
        for cols, pre in zip(data_columns, pre_compute) if not pre]

    # investigate if the post functions depends on other post functions
    one_at_the_time = False
    is_in = [any([idx in indices for idx in column_index_post])
             for indices in column_indices]
    if any(is_in):
        order = [i if not elem else len(is_in)+i for i, elem in enumerate(is_in)]
        # max_column_indices = [max(elem) for elem in column_indices]
        column_indices = [x for y, x in sorted(zip(order, column_indices))]
        column_index_post = [x for y, x in sorted(zip(order, column_index_post))]
        one_at_the_time = True

    # a function to get the value of each obj in the array
    get_value = np.vectorize(lambda obj: obj['value'].value, otypes=[float])

    # Calculate the post-functions
    if one_at_the_time:
        for i, idx in zip(column_index_post, column_indices):
            results_array[i, :] = functions[i](*get_value(results_array[idx, :]))
    else:
        post_functions = [func for func, pre in zip(functions, pre_compute) if not pre]
        results_array[column_index_post, :] = [
            func(*get_value(results_array[idx, :]))
            for func, idx in zip(post_functions, column_indices)
        ]
    return results_array


@recordable
def pivot_result_table(
        name: str,
        dataset: data.DataSet,
        column_names: Sequence[str],
        functions: Sequence[Callable],
        data_columns: Sequence[Sequence[str]],
        pre_compute: Sequence[bool],
        segmentation: Optional[data.Segmentation] = None,
) -> ResultTable:
    if dataset.observations == 0:
        result_list = [ScalarResult("NAN", np.nan)] * len(column_names)
        [elem.lower() for elem in column_names].index('observations')
        result_list[0] = ScalarResult("COUNT", 0)
        return ResultTable(
            name=name,
            row_names=['Total'],
            column_names=list(column_names),
            results=np.array([result_list])
        )
    total_results_array = calc_row_result(
        data_input=dataset, column_names=column_names,
        functions=functions, data_columns=data_columns, pre_compute=pre_compute)
    if segmentation:
        # if segmentation, calculate result array for the segments
        # (similar to result for dataset)
        results_array = calc_row_result(
            data_input=segmentation, column_names=column_names,
            functions=functions, data_columns=data_columns, pre_compute=pre_compute)
        row_names = [segment.segment_id for segment in segmentation.segments]
        return ResultTable(
            name=name,
            row_names=list(row_names + ['Total']),
            column_names=list(column_names),
            results=np.concatenate((results_array, total_results_array), axis=1).T
        )
    else:
        return ResultTable(
            name=name,
            row_names=['Total'],
            column_names=list(column_names),
            results=total_results_array.T
        )


@recordable
def concentration_test_result_table(
        dataset: data.DataSet,
        column_name_exposure: str,
        segmentation: Optional[data.Segmentation] = None,
) -> ResultTable:

    if segmentation is not None:
        column_name_observation = segmentation.by
    else:
        column_name_observation = column_name_exposure

    inputs = [
        ['Observations', metric.count, column_name_observation, True],
        ['Observations Concentration', metric.relative_frequency, 'Observations', False],
        ['Exposure', metric.sum, column_name_exposure, True],
        ['Exposure Concentration', metric.relative_frequency, 'Exposure', False]
    ]
    return pivot_result_table(
        name='CONCENTRATION TEST',
        dataset=dataset,
        segmentation=segmentation,
        column_names=[row[0] for row in inputs],
        functions=[row[1] for row in inputs],
        data_columns=[row[2] for row in inputs],
        pre_compute=[row[3] for row in inputs]
        )


@recordable
def herfindahl_index_table(
        result_table: ResultTable,
) -> ResultTable:

    column_names = ['Observations Concentration', 'Exposure Concentration']
    sub_columns = result_table.get_column_results(column_names=column_names, value=True).T
    row_names = list(result_table.row_names)
    if 'Total' in row_names:
        row_names.remove('Total')
        sub_columns = sub_columns[:, :-1]

    herfindahl_indices = np.array([
        [metric.herfindahl_index(relative_frequencies=sub_columns[0, ], amber=0.2, red=0.25),
         metric.herfindahl_index(relative_frequencies=sub_columns[1, ], amber=0.2, red=0.25)]
        ])

    return ResultTable('Herfindahl Index table', ['Herfindahl Index'], column_names, herfindahl_indices)


@recordable
def pd_back_test_result_table(
        dataset: data.DataSet,
        column_name_exposure: str,
        column_name_defaulted: str,
        column_name_pd: str,
        segmentation: Optional[data.Segmentation] = None,
        red=0.05,
        amber=0.05 * 2
) -> ResultTable:

    if segmentation is not None:
        column_name_observation = segmentation.by
    else:
        column_name_observation = column_name_exposure

    inputs = [
        ['Observations', metric.count, column_name_observation, True],
        ['Exposure', metric.sum, column_name_exposure, True],
        ['Defaulted', metric.sum, column_name_defaulted, True],
        ['PD', metric.mean_value, column_name_pd, True],
        ['DF', np.vectorize(metric.ratio), ('Defaulted', 'Observations'), False],  # ['DF', metric.mean_value, column_name_defaulted, True],
        ['PD-DF', np.vectorize(metric.difference), ('PD', 'DF'), False],
        ['(PD-DF)/DF', np.vectorize(metric.relative_difference), ('PD', 'DF'), False],
        ['Jeffreys test (H0: PD ≥ DF)', np.vectorize(partial(metric.jeffreys_test, red=red, amber=amber)),
         ('Observations', 'Defaulted', 'PD'), False],
    ]

    result_table = pivot_result_table(
        name='PD BACK TEST',
        dataset=dataset,
        segmentation=segmentation,
        column_names=[row[0] for row in inputs],
        functions=[row[1] for row in inputs],
        data_columns=[row[2] for row in inputs],
        pre_compute=[row[3] for row in inputs]
        )
    result_table._inputs = inputs
    return result_table


# TODO: this is not used at the moment. Not sure if it will
@recordable
def jeffreys_test_table(
        result_table: ResultTable,
) -> ResultTable:

    column_names = ['Observations', 'Defaulted', 'PD']
    sub_columns = result_table.get_column_results(column_names=column_names, value=True)
    row_names = list(result_table.row_names)

    jeffreys_tests = np.array([
        [metric.jeffreys_test(n=obs, x=defaults, applied_p=pd)]
        for (obs, defaults, pd) in sub_columns])

    return ResultTable('Jeffreys test table', row_names, ['Jeffreys test'], jeffreys_tests)


# TODO: this is not used at the moment (in local site)
@recordable
def overwrites_result_table(
        dataset: data.DataSet,
        column_name_overwrites: str,
        segmentation: Optional[data.Segmentation] = None,
        ) -> ResultTable:

    inputs = [
        ['Nr. of overwrites', metric.sum, column_name_overwrites, True],
        ['Observations', metric.count, column_name_overwrites, True],
        ['Ratio', np.vectorize(metric.ratio), ('Nr. of overwrites', 'Observations'), False]
    ]
    return pivot_result_table(
        name='OVERWRITES',
        dataset=dataset,
        segmentation=segmentation,
        column_names=[row[0] for row in inputs],
        functions=[row[1] for row in inputs],
        data_columns=[row[2] for row in inputs],
        pre_compute=[row[3] for row in inputs]
        )
