from typing import Sequence, Tuple, TypeVar, Union

import numpy as np
from scipy.stats import norm

T = TypeVar('T')
Vector = Union[Sequence[T], np.ndarray]


def relative_frequency(
        v: Vector[float],
        segment):
    """
    Calculates the relative frequency for each state in segment. i.e. if segment is
    rating grades it is the relative frequency of rating grades. The Vector v denotes
    the measure. i.e. if v is number of customers in each rating grade it is
    number-weighted relative frequencies, or if v is exposure in each rating grade it is
    exposure-weighted relative frequencies


    Parameters
    ----------
    v (of the segment):  vector
    segment: vector
    """

    k = len(segment)
    if k != len(np.unique(segment)):
        raise ValueError('values in segment is not all unique\n'
                         f"len(segment) = {k} != {len(np.unique(segment))} ="
                         f"len(np.unique(segment))")

    if len(v) != k:
        raise ValueError(
            'v and segment is not of same shape is not all unique\n'
            f"len(segment) = {k} != {len(np.unique(segment))} ="
            f"len(np.unique(segment))")

    return v/np.sum(v)


def herfindahl_index(
        relative_frequencies: Vector[float],
        segment: Vector,
) -> Tuple[float, float]:
    """
    Calculate the Herfindahl Index and its Coefficient of Variation

    Parameters
    ----------
    relative_frequencies (of the segment):  vector
    segment: vector
    """

    k = len(segment)
    if k != len(np.unique(segment)):
        raise ValueError('values in segment is not all unique\n'
                         f"len(segment) = {k} != {len(np.unique(segment))} ="
                         f"len(np.unique(segment))")

    if len(relative_frequencies) != k:
        raise ValueError('relative_frequencies and segment is not of same shape is not '
                         'all unique\n'
                         f"len(segment) = {k} != {len(np.unique(segment))} ="
                         f"len(np.unique(segment))")
    k = len(segment)
    # coefficient of variation squared
    cv_squared = k * np.sum((relative_frequencies - 1 / k)**2)
    # Herfindahl Index
    h_i = 1 + np.log((cv_squared + 1) / k)/np.log(k)

    return h_i, np.sqrt(cv_squared)


def herfindahl_index_test(
        cv_initial,
        cv_current,
        segment: Vector,
):
    """
    Comparison of the Herfindahl Index at the beginning of the relevant observation
    period and the Herfindahl Index at the time of the initial validation during
    development via hypothesis testing based on a normal approximation assuming a
    deterministic Herfindahl Index at the time of the model's development. The null
    hypothesis of the test is:
    H0: current Herfindahl Index ≤  initial Herfindahl Index
    (it is upper/right-tailed (one tailed) Z-test)
    """
    k = len(segment)
    if k != len(np.unique(segment)):
        raise ValueError('values in segment is not all unique\n'
                         f"len(segment) = {k} != {len(np.unique(segment))} ="
                         f"len(np.unique(segment))")

    cv_current_squared = cv_current**2
    numerator = np.sqrt(k-1)*(cv_current - cv_initial)
    denominator = np.sqrt(cv_current_squared*(0.5+cv_current_squared))
    test_statistic = numerator/denominator
    p_value = 1 - norm.cdf(numerator/denominator, loc=0, scale=1)

    return test_statistic, p_value
