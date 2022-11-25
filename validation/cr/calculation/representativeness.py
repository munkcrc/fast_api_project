from typing import Literal

import numpy as np

from typing import Dict, Optional, Sequence, Tuple, TypeVar, Union
T = TypeVar('T')
Vector = Union[Sequence[T], np.ndarray]


def psi_sub_term(a: float, b: float) -> float:
    """
    Calculates a sum term (summand or addend) for the sum to be a PSI score:
    The formula for PSI score for a variable divided into K mutually exclusive
    buckets (categories) is
        psi_numerical = sum( (a_i - b_i) * ln(a_i / b_i) for i in K),
    where a_i and b_i is the relative frequency of the value in bucket i for the
    first dataset and second dataset respectively.
    """
    if a == b:
        return 0
    else:
        if a == 0:
            a = 0.0001
        if b == 0:
            b = 0.0001
        return (a - b) * np.log(a / b)


def psi_numerical(
        a: Vector[float],
        b: Vector[float],
        buckets: Union[int, Sequence[float]] = 5,
        bucket_type: Optional[Literal['bins', 'quantiles']] = 'bins',
) -> Tuple[float, Dict]:  # np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
        Calculate the PSI for a single variable which is of a numerical type.
    Args:
        a: vector of samples a
        b: vector of samples b, same size as a
        buckets : int or sequence of scalars
            If `buckets` is an int, it defines the number of bins equal-with
            bins based quantiles of a or ranges of a (chosen by bucket_type).
            If `buckets` is a sequence, it defines a monotonically increasing array of
            bin edges, including the rightmost edge, allowing for non-uniform bin
            widths. In other words, if `buckets` is:

                [1, 2, 3, 4]

            then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
            the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
            *includes* 4.
        bucket_type: type of strategy for creating buckets,
            bins splits into even splits, quantiles splits into quantile buckets
    Returns:
       a tuple with:
            the PSI value,
            a dictionary with intermediate results:
                bin edges used for the buckets
                relative frequency of the buckets for vector a and b,
                PSI sum term (summands)
                a dictionary with relevant information for not finite terms
    """
    # split input vectors up based inf and nan and numbers
    def split_up_based_on_not_finite(v: Vector):
        mask = np.isfinite(v)
        v_not_finite = v[~mask]
        v_missing = v_not_finite[np.isnan(v_not_finite)]
        v_neg_inf = v_not_finite[v_not_finite < 0]
        v_pos_inf = v_not_finite[0 < v_not_finite]
        return v[mask], v_missing, v_neg_inf, v_pos_inf

    a_clean, a_missing, a_neg_inf, a_pos_inf = split_up_based_on_not_finite(a)
    b_clean, b_missing, b_neg_inf, b_pos_inf = split_up_based_on_not_finite(b)
    len_a = max(len(a), 1)
    len_b = max(len(b), 1)
    non_finite = {'relative_frequency': {
        'missing': {'a': len(a_missing) / len_a, 'b': len(b_missing) / len_b},
        'neg_inf': {'a': len(a_neg_inf) / len_a, 'b': len(b_neg_inf) / len_b},
        'pos_inf': {'a': len(a_pos_inf) / len_a, 'b': len(b_pos_inf) / len_b}
    }}
    non_finite['psi_summands'] = {
        'missing': psi_sub_term(**non_finite['relative_frequency']['missing']),
        'neg_inf': psi_sub_term(**non_finite['relative_frequency']['neg_inf']),
        'pos_inf': psi_sub_term(**non_finite['relative_frequency']['pos_inf'])
    }

    if a_clean.size > 0 and b_clean.size > 0:
        if isinstance(buckets, Sequence):
            bin_edges = np.array(buckets)
        elif isinstance(buckets, np.ndarray):
            bin_edges = buckets
        elif bucket_type == 'quantiles':
            bin_edges = np.unique(
                np.percentile(a_clean, np.arange(0, buckets + 1) / buckets * 100))
        else:  # bucket_type == 'bins':
            bin_edges = np.linspace(np.min(a_clean), np.max(a_clean), buckets + 1)

        bin_edges[0] = min(np.min(a_clean), np.min(b_clean))
        bin_edges[-1] = max(np.max(a_clean), np.max(b_clean))
    else:
        bin_edges = np.array([])

    percents = np.array([
        np.histogram(a_clean, bin_edges)[0] / len_a,
        np.histogram(b_clean, bin_edges)[0] / len_b
    ])
    psi_summands = np.array([psi_sub_term(a, b) for (a, b) in percents.T])
    psi_total = sum(psi_summands) + sum([*non_finite['psi_summands'].values()])
    dict_intermediate = {
        'bin_edges': bin_edges,
        'relative_frequency': {'a': percents[0, :], 'b': percents[1, :]},
        'psi_summands': psi_summands,
        'non_finite': non_finite
    }
    if a_clean.size == 0 or b_clean.size == 0:
        psi_total = np.nan
    return psi_total, dict_intermediate


def psi_categorical(
        a: Vector,
        b: Vector,
) -> Tuple[float, Dict]:  # Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
        Calculate the PSI for a single variable which is of a categorical type.
        The number of buckets corresponds to number of unique values from the variable.
    Args:
        a: vector of samples a
        b: vector of samples b, same size as a
    Returns:
       a tuple with:
            the PSI value,
            a dictionary with intermediate results:
                the discrete buckets (unique values)
                relative frequency of the buckets for vector a and b,
                PSI sum term (summands)
    """
    a_unique, a_count = np.unique(a, return_counts=True)
    b_unique, b_count = np.unique(b, return_counts=True)
    if a_unique.size == b_unique.size > 0 and np.all(a_unique == b_unique):
        percents = np.array([
            a_count / np.sum(a_count),
            b_count / np.sum(b_count)
        ])
        psi_summands = np.array([psi_sub_term(a, b) for (a, b) in percents.T])
        dict_intermediate = {
            'buckets': a_unique,
            'relative_frequency': {'a': percents[0, :], 'b': percents[1, :]},
            'psi_summands': psi_summands}
        return sum(psi_summands), dict_intermediate
    else:
        buckets = np.unique(np.concatenate([a, b]))
        psi_value, dict_intermediate = psi_categorical_grouped(a, b, buckets)
        dict_intermediate['buckets'] = buckets
        return psi_value, dict_intermediate


def psi_categorical_grouped(
        a: Vector,
        b: Vector,
        buckets: Sequence[Sequence[str]],
) -> Tuple[float, Dict]:  # Tuple[float, np.ndarray, np.ndarray]:
    """
        Calculate the PSI for a single variable which is of a categorical type using
        the input buckets.
    Args:
        a: vector of samples a
        b: vector of samples b, same size as a
        buckets: a sequence/vector of sequence of categorical buckets. i.e.
            [['a','c'],['b','d']
    Returns:
       a tuple with:
            the PSI value,
            a dictionary with intermediate results:
                PSI sum term (summands)
                relative frequency of the buckets for vector a,
                relative frequency of the buckets for vector b,
    """
    if a.size > 0 and b.size > 0:
        counts = np.array([
            [np.sum(np.in1d(a, bucket)), np.sum(np.in1d(b, bucket))] for bucket in buckets
        ]).T
        percents = np.array([
            counts[0, :] / np.sum(counts[0, :]),
            counts[1, :] / np.sum(counts[1, :])
        ])
        psi_summands = np.array([psi_sub_term(a, b) for (a, b) in percents.T])
        dict_intermediate = {
            'relative_frequency': {'a': percents[0, :], 'b': percents[1, :]},
            'psi_summands': psi_summands}
        return sum(psi_summands), dict_intermediate
    else:
        return np.nan, {
            'relative_frequency': {'a': np.array([]), 'b': np.array([])},
            'psi_summands': np.array([])
        }


def get_categorical_buckets(
        a: Vector,
        b: Vector,
        buckets: int = 5,
        bucket_type: Optional[Literal['bins', 'quantiles']] = 'bins',
        order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
        Calculate the buckets used as input to calculate PSI for a categorical variable.
    Args:
        a: vector of samples a
        b: vector of samples b, same size as a
        buckets : defines the number of bins equal-with
            bins based quantiles of a or ranges of a (chosen by bucket_type).
        bucket_type: type of strategy for creating buckets,
            bins splits into even splits, quantiles splits into quantile buckets
        order: specify the order of the categorical values. Is None, the order used is
            the one specified by np.unique (alphabetic or numerical)
    Returns:
       buckets
    """

    if order is None:
        # if no order, we order by the sorting in np.unique
        order = np.unique(np.concatenate([a, b]))
        idx_order = np.arange(0, len(order))

        idx_a = np.searchsorted(order, a)
        order_a = idx_order[idx_a]
        idx_b = np.searchsorted(order, b)
        order_b = idx_order[idx_b]
    else:
        order = np.array(order)
        map_dict = {k: i for i, k in enumerate(order)}
        order_a = np.array([map_dict[key] for key in a])
        order_b = np.array([map_dict[key] for key in b])

        idx_order = np.arange(0, len(order))

    order_a_b = np.concatenate([order_a, order_b])
    if bucket_type == 'quantiles':
        bin_edges = np.unique(
            np.percentile(order_a_b, np.arange(0, buckets + 1) / buckets * 100))
    else:  # bucket_type == 'bins':
        bin_edges = np.linspace(np.min(order_a_b), np.max(order_a_b), buckets + 1)

    bin_edges[-1] += 1

    belong_to_bin = np.digitize(idx_order, bin_edges)
    buckets = np.array([
        list(order[belong_to_bin == elem]) for elem in np.unique(belong_to_bin)],
        dtype=order.dtype)

    # Is this more clean? :
    # buckets = np.split(
    #   order,
    #   np.unique(np.digitize(idx_order, bin_edges), return_index=True)[1][1:]
    #   )

    return buckets


def psi_for_matrix(
        a: np.ndarray,
        b: np.ndarray,
        buckets: int = 10,
        bucket_type: Literal['bins', 'quantiles'] = 'bins',
        axis: int = 0) -> np.ndarray:
    """
        Measure PSI between samples a and b for several variables
    Args:
       a: numpy matrix of samples a
       b: numpy matrix of samples b, same size as a is expected
       buckets: number of buckets
       bucket_type: type of strategy for creating buckets,
            bins splits into even splits, quantiles splits into quantile buckets
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi_numerical values for each variable
    """
    if len(a.shape) == 1:
        psi_values = np.empty(len(a.shape))
    else:
        psi_values = np.empty(a.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi_numerical(a, b, buckets, bucket_type)
        elif axis == 0:
            psi_values[i] = psi_numerical(a[:, i], b[:, i], buckets, bucket_type)
        elif axis == 1:
            psi_values[i] = psi_numerical(a[i, :], b[i, :], buckets, bucket_type)

    return psi_values