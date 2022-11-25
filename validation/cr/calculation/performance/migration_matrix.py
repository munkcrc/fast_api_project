from typing import Sequence, TypeVar, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

T = TypeVar('T')
Vector = Union[Sequence[T], np.ndarray]


def migration_matrix(
        start: Vector[float],
        end: Vector[int],
        drop_nan=True,
        order=None,
        include_all=False,
):
    """
    angiv start state og end state vektor og få ud migrations matricer:
        en som angiver antal migrationer fra start statie til slut statie
        en som angiver sansynligheden migration fra start statie til slut statie
    start og end skal have samme længde.

    optional (drop_nan):
        hvis True, sorter væk de (start, end)-par som indeholder en eller to np.nan.
        hvis False, inkluder de (start, end)-par som indeholder np.nan, og inkluder den
        i order hvis order ikke er None

    optional (order): angiv rækkefølgen på de stadier som matricen inderholder.
    Hvis man angiver en stadie som ikke ingår bliver den ignoreret (medmindre include_all=True)
    Hvis man ikke angiver en stadie som ellers er i start og end, så forsvinder den.

    optional (include_all), Hvis der er en stadie i order, som ikke indgår i start
        eller end så inkluder den. Den rækker bliver 0 i count, og np.nan i prop


    """
    if len(start) != len(end):
        raise ValueError('start and end are not of same length'
                         f"\n{' '*len('ValueError:')} "
                         f"len(start)={len(start)}, len(end)={len(end)}")

    if drop_nan:
        df = pd.DataFrame(data={'START': start, 'END': end}).dropna()
        if not np.issubdtype(df['START'].dtype, np.number):
            df = df[~np.isin(df['START'], ('nan', 'None', '', '<NA>'))]
        if not np.issubdtype(df['END'].dtype, np.number):
            df = df[~np.isin(df['END'], ('nan', 'None', '', '<NA>'))]
    else:
        df = pd.DataFrame(data={'START': start, 'END': end}).fillna('nan')
        if (not np.issubdtype(df['START'].dtype, np.number)) or (
                not np.issubdtype(df['END'].dtype, np.number)):
            df.replace(
                to_replace=('nan', 'None', '', '<NA>'), value='nan', inplace=True)

    df_count = pd.crosstab(df['START'], df['END'], normalize=False, dropna=False)

    if order is not None:
        if not drop_nan:
            if 'nan' in df_count.index or 'nan' in df_count.columns:
                if all(pd.notna(order)) and 'nan' not in order:
                    order = list(order)
                    order.append('nan')
        if include_all:
            not_in_row = [elem for elem in order if elem not in df_count.index]
            not_in_col = [elem for elem in order if elem not in df_count.columns]
            for elem in not_in_row:
                df_count = df_count.append(pd.Series(0, index=df_count.columns, name=elem))
            for elem in not_in_col:
                df_count[elem] = pd.Series(0, index=df_count.index)

        # use list comprehension instead of order[np.isin(order, row_state)],
        # since np.isin differ between python int and numpy int: np.isin(1, np.int(1))
        row_order = [elem for elem in order if elem in df_count.index]
        col_order = [elem for elem in order if elem in df_count.columns]
        df_count = df_count.loc[row_order, col_order]

    migration_count = df_count.values

    migration_prob = migration_count/np.clip(
        migration_count.sum(axis=1)[:, None], 1, None)

    # replace rows with all zeros to be np.nan
    if include_all:
        nan_mask = migration_count.sum(axis=1) == 0
        if np.any(nan_mask):
            nans = np.ones_like(nan_mask, dtype=np.float64)
            nans[nan_mask] = np.nan
            migration_prob = migration_prob*nans[:, None]

    return (
        migration_prob,
        migration_count,
        df_count.index.values,
        df_count.columns.values
    )


def matrix_weighted_bandwidth(migration_count: np.ndarray, upper=True):
    """
    The objective is to analyse the migration of customers across
    rating grades during the relevant observation period.

    The function return a statistics (𝑀𝑊𝐵) that are calculated in order
    to summarise upgrades and downgrades, i.e. respective values above and below the
    diagonal of the migration matrix. (upper determines is it is upper up or down)

    notice that N_i * p_{i,j} = c_{i,j}

    formula:
    upper MWB =
        [ sum_{i=1}^{k-1} ( sum_{j=i+1}^{k} |i - j| * c_{i,j} ) ] / M_norm_upper

    notice that the summation means that it is a upper triangle matrix (with zero in
    the diagonal), that should be summed.

    lower MWB =
        ( sum_{i=2}^{k} ( sum_{j=1}^{i-1} |i - j| * c_{i,j} ) ) / M_norm_lower

    notice that the summation means that it is a lower triangle matrix (with zero in
    the diagonal), that should be summed.

    """
    if len(migration_count.shape) != 2:
        raise ValueError('migration_count is not a 2-dimensional matrix:\n'
                         f"migration_count.shape = {migration_count.shape}")

    if migration_count.shape[0] != migration_count.shape[1]:
        raise ValueError('migration_count is not a square matrix '
                         '(a matrix with the same number of rows and columns):\n'
                         f"migration_count.shape={migration_count.shape}")

    k = migration_count.shape[0]

    # a (K x K) matrix with ones in the lower corner (zero in diagonal)
    # lower: from 2 to k
    triangle_matrix = np.tri(k, k, -1, dtype=np.int64)
    if upper:
        # if upper we transpose to get an upper triangle_matrix
        # upper: from 1 to k-1
        triangle_matrix = triangle_matrix.T

    # np.mgrid is np.arange in multi-dimension.
    # obs: the end point is not included
    row_and_column_indices = np.mgrid[1:k+1, 1:k+1]
    row_indices_matrix = row_and_column_indices[0]
    column_indices_matrix = row_and_column_indices[1]

    # the "|i-j|" part for each row i and column j
    abs_matrix = np.abs(row_indices_matrix - column_indices_matrix) * triangle_matrix

    # the "|i - j| * c_{i,j}" part (or "|i - j| * N_i * p_{i,j}") for each
    # row i and column j, summed together to the numerator
    numerator = np.sum(abs_matrix*migration_count)

    def m_norm():
        """
        m_norm (matrix_norm or migration_norm?) is used to calculate the
        matrix_weighted_bandwidth()

        notice that N_i * p_{i,j} = c_{i,j}

        formula:
        M_norm_upper =
            sum_{i=1}^{k-1} [ max(|i-k|,|i-1|) * ( sum_{j=i+1}^{k} c_{i,j} ) ]

        notice that the summation means that it is a upper triangle matrix (with zero in
        the diagonal), that should be summed.

        M_norm_lower =
            sum_{i=2}^{k} [ max(|i-k|,|i-1|) * ( sum_{j=1}^{i-1} c_{i,j} ) ]

        notice that the summation means that it is a lower triangle matrix (with zero in
        the diagonal), that should be summed.

        """
        # obs: the end point is not included
        row_indices = np.arange(1, k + 1, dtype=np.int64)[:, None]

        # the "max(|i-k|,|i-1|)" part for each row i
        max_vector = np.max(
            np.concatenate((
                np.abs(row_indices - k), np.abs(row_indices - 1)
            ), axis=1)
            , axis=1)[:, None]

        # the "sum_{j=i+1}^{k} c_{i,j}" part for each row i and column j,
        # summed together
        inner_sum = np.sum(migration_count * triangle_matrix, axis=1)[:, None]

        # sum it all together
        return np.sum(max_vector * inner_sum)

    # return the mwb
    return numerator/m_norm()


def stability_of_migration_test(
        migration_prob: np.ndarray,
        migration_count: np.ndarray):
    """
    The objective is to verify the monotonicity of off-diagonal transition frequencies in
    the migration matrix by means of z-tests, thereby identifying possible portfolio shifts.

    Consider the entries in the migration matrix corresponding to the status 'rating
    grades 1 to K' at the beginning and at the end of the relevant observation period on
    the basis of the number of customers (N) as defined in points (g) and (h)(i) of Section
    2.5.1. The fact that rating migrations follow a multinomial distribution can be
    exploited by pairwise z-tests exploiting the asymptotic normality of the test statistic.
    Let p_{i,j} denote the (observed) relative frequency of transition (i.e. the relative
    frequency of customer migrations) between rating grade i (at the beginning of the
    relevant observation period) and rating grade j (at the end of that observation
    period). The null hypothesis of the tests is either
    H0: p_{i,j} ≥ p_{i,j-1} or      (lower)
    H0: p_{i,j-1} ≥ p_{i,j}         (upper)
    (lower/left-tailed)
    depending on whether the {i,j} entry in the migration matrix is below or above the
    main diagonal. i.e. for a migration matrix with rating grades 1 to K = 4 we test

     | p_{1,1} ≥ p_{1,2} ≥ p_{1,3} ≥ p_{1,4} |
     | p_{2,1} ≤ p_{2,2} ≥ p_{2,3} ≥ p_{2,4} |
     | p_{3,1} ≤ p_{3,2} ≤ p_{3,3} ≥ p_{3,4} |
     | p_{4,1} ≤ p_{4,2} ≤ p_{4,3} ≤ p_{4,4} |

     which consist of K*(K-1) (lower/left-tailed) tests (one for each inequality)

     for for 1 ≤ j < i (lower off-diagonal)
     a = p_{i,j}
     b = p_{i,j+1}
     z_{i,j} = (b - a) / sqrt( (a(1-a) + b(1-b) + 2bc)/N_i )

     for for i < j ≤ K (upper off-diagonal)
     a = p_{i,j-1}
     b = p_{i,j}
     z_{i,j} = (a - b) / sqrt( (b(1-b) + a(1-a) + 2ba)/N_i )

     """
    count_vector = migration_count.sum(axis=1)[:, None]
    r = migration_prob.shape[0]
    k = migration_prob.shape[1]
    left = migration_prob[:, 0:k - 1]
    right = migration_prob[:, 1:k]
    denominator = np.sqrt(
        (left * (1 - left) + right * (1 - right) + 2 * left * right) / count_vector)

    # for the lower triangle the numerator is p_{i,j+1} - p_{i,j}  (right - left)
    # for the upper triangle the numerator is p_{i,j-1} - p_{i,j}  (left - right)
    # notice that (left - right) = (-1)*(right - left)
    triangle_matrix = np.tri(r, k - 1, -1, dtype=np.int64)
    triangle_matrix[triangle_matrix == 0] = -1
    z_matrix = (right - left) * triangle_matrix / denominator
    # TODO: we should have a Z-test function instead that evaluate a z test statistic,
    #  given if it is a lower/left-tailed, upper/right-tailed tests or two-tailed tests.
    #  Perhaps this make sense when we hypothesis classes are made in cr.tesing.result
    p_matrix = norm.cdf(z_matrix, loc=0, scale=1)

    mask_leq = triangle_matrix == -1
    mask_geq = triangle_matrix == 1
    h0_matrix = np.empty(shape=triangle_matrix.shape, dtype=object)
    h0_matrix[mask_leq] = '≥'
    h0_matrix[mask_geq] = '≤'

    return p_matrix, z_matrix, h0_matrix


# migration_prob = rt_all.to_dataframe(attribute="value", value=True).values[:, :-1]
# migration_count = rt_all.to_dataframe(attribute="count", value=True).values[:, :-1]
# index_default_column = migration_count.shape[1]-1
def stability_of_migration_column(
        migration_prob: np.ndarray,
        migration_count: np.ndarray,
        index_column: int = None):
    """
    https://stats.stackexchange.com/questions/113602/test-if-two-binomial-distributions-are-statistically-different-from-each-other
    let D be the index of the column in the migration matrix. The tests are:
    H0: p_{i,D} ≤ p_{i+1,D} for 0 ≤ i < D
     a = p_{i,D}
     b = p_{i+1,D}
     z_{i,j} = (b - a) / sqrt( p_hat(1-p_hat)*(1/n_i + 1/n_{i+1}) )
     p_hat = (n_i * p_{i,D} + n_{i+1} * p_{i+1,D})/(n_i+n_{i+1})
           = (x_i + x_{i+1}) / (n_i+n_{i+1})

    (lower/left-tailed)
    """
    if index_column is None:
        index_column = migration_prob.shape[1]-1

    count_vector = migration_count.sum(axis=1)[:, None]
    prob_vector = migration_prob[:, index_column][:, None]
    x_vector = migration_count[:, index_column][:, None]

    p_hat_vector = (x_vector[:-1] + x_vector[1:])/(count_vector[:-1]+count_vector[1:])

    denominator = np.sqrt(
        p_hat_vector * (1 - p_hat_vector) * (1/count_vector[:-1] + 1/count_vector[1:])
    )

    numerator = prob_vector[1:] - prob_vector[:-1]

    test_statistics = numerator/denominator

    norm.cdf(test_statistics, loc=0, scale=1)

    h0_matrix = np.empty(shape=test_statistics.shape, dtype=object)
    h0_matrix.fill('^')

    return test_statistics, h0_matrix
