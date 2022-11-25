import numpy as np
import cr.calculation as calculate
from cr.automation import recordable
from cr.testing.result import ScalarRAGResult, Result, ResultTable, ScalarResult
import cr.testing.metric.hypothesis as hypothesis
from scipy.stats import norm
from typing import List, Optional


@recordable
def migration_matrix(
        start, end, drop_nan=True, order=None, include_all=False) -> ResultTable:
    (migration_prob, migration_count, row_names, column_names) = calculate.migration_matrix(
        start=start,
        end=end,
        drop_nan=drop_nan,
        order=order,
        include_all=include_all,
    )

    def _scalar_result_with_count(p_value, c_value):
        return ScalarResult(name='Migration Matrix Entry', value=p_value).add_outputs({
            "count": c_value
        })

    results = np.array(
        [[_scalar_result_with_count(p, c) for p, c in zip(p_row, c_row)]
         for p_row, c_row in zip(migration_prob, migration_count)]
    )
    return ResultTable(
        'Migration Matrix',
        row_names=list(row_names),
        column_names=list(column_names),
        results=results,
    )


@recordable
def matrix_stability(migration: ResultTable, ignore_states: Optional[List[str]] = None, red=0.05, amber=0.05*2):
    """
    H0: p_{i,j-1} ≥ p_{i,j} (for upper diagonal matrix)
    H0: p_{i,j-1} ≤ p_{i,j} (for lower diagonal matrix)
    """
    if ignore_states is None:
        ignore_states = []

    df_migration_p = migration.to_dataframe(attribute="value", value=True).drop(
        index=ignore_states, columns=ignore_states, errors='ignore')
    migration_prob = df_migration_p.values

    df_migration_c = migration.to_dataframe(attribute="count", value=True).drop(
        index=ignore_states, columns=ignore_states, errors='ignore')
    migration_count = df_migration_c.values

    p_matrix, z_matrix, h0_matrix = calculate.stability_of_migration_test(
        migration_prob=migration_prob,
        migration_count=migration_count)
    col_names = df_migration_p.columns
    cols_to_test = list(zip(col_names[:-1], col_names[1:]))

    def _left_tailed_rag_entry(p, z, h0, c):
        if np.any(np.isnan([p, z])):
            return ScalarResult(
                name="Stability Migration Matrix Entry",
                value=np.nan)
        else:
            return hypothesis.left_tailed_rag(
                name="Stability Migration Matrix Entry",
                test_statistic=z,
                pdf=norm.pdf,
                cdf=norm.cdf,
                cdf_inverse=norm.ppf,
                p_value=p,
                red=red,
                amber=amber).add_outputs({
                    "h0": h0,
                    "figure_title": f"H0: {c[0]} {h0} {c[1]}"})

    results = np.array(
        [[_left_tailed_rag_entry(p, z, h0, (c1, c2)) for p, z, h0, (c1, c2) in
          zip(p_row, z_row, h0_row, cols_to_test)]
         for p_row, z_row, h0_row, in zip(p_matrix, z_matrix, h0_matrix)]
    )
    return ResultTable(
        'Stability Migration Matrix',
        row_names=list(df_migration_p.index),
        column_names=[f"{c1} vs. {c2}" for c1, c2 in cols_to_test],
        results=results,
    ).add_outputs({"Migration Matrix": migration})


@recordable
def matrix_stability_column(migration: ResultTable, index_column: int = None, red=0.05, amber=0.05*2):
    """
    let D be the index of the column in the migration matrix. The tests are:
    H0: p_{i,D} ≤ p_{i+1,D} for 0 ≤ i < N
    """

    df_migration_p = migration.to_dataframe(attribute="value", value=True)
    migration_prob = df_migration_p.values

    df_migration_c = migration.to_dataframe(attribute="count", value=True)
    migration_count = df_migration_c.values

    test_statistics, h0_matrix = calculate.stability_of_migration_column(
        migration_prob=migration_prob,
        migration_count=migration_count,
        index_column=index_column)

    def _left_tailed_rag_entry(z, h0):
        if np.isnan(z):
            return ScalarResult(
                name="Stability Migration Matrix Entry",
                value=np.nan)
        else:
            return hypothesis.left_tailed_rag(
                name="Stability Migration Matrix Entry",
                test_statistic=z,
                pdf=norm.pdf,
                cdf=norm.cdf,
                cdf_inverse=norm.ppf,
                red=red,
                amber=amber).add_outputs({"h0": h0})

    results = np.array(
        [[_left_tailed_rag_entry(z, h0) for z, h0 in zip(z_row, h0_row)]
         for z_row, h0_row, in zip(test_statistics, h0_matrix)]
    )
    if index_column is None:
        col_name = migration.column_names[-1]
    else:
        col_name = migration.column_names[index_column]

    row_name = migration.row_names
    return ResultTable(
        'Stability Migration Column',
        row_names=[f"{x1} ≤ {x2}" for x1, x2 in zip(row_name[:-1], row_name[1:])],
        column_names=[col_name],
        results=results,
    ).add_outputs({"Migration Matrix": migration})

