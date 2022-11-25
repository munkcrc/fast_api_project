from typing import Tuple

import numpy as np

from typing import Union, Sequence, TypeVar
T = TypeVar('T')
Vector = Union[Sequence[T], np.ndarray]


def outlier_tukey_fences(
        v: Vector[float],
        k: float = 1.5) -> Tuple[float, float, float, float, float]:
    q_1, q_3 = np.nanquantile(a=v, q=[0.25, 0.75], interpolation='midpoint')
    iqr = q_3 - q_1
    lower = q_1 - k*iqr
    upper = q_3 + k*iqr
    nr_below = (v < lower).sum()
    nr_above = (upper < v).sum()
    nr_of_outliers = nr_below + nr_above

    return nr_of_outliers, nr_below, nr_above, lower, upper


def outlier_asymmetric_tukey_fences(
        v: Vector[float],
        k: float = 1.5) -> Tuple[float, float, float, float, float]:
    q_1, q_2, q_3 = np.nanquantile(a=v, q=[0.25, 0.5, 0.75], interpolation='midpoint')
    lower = q_1 - 2*k * (q_2-q_1)
    upper = q_3 + 2*k * (q_3-q_2)
    nr_below = (v < lower).sum()
    nr_above = (upper < v).sum()
    nr_of_outliers = nr_below + nr_above

    return nr_of_outliers, nr_below, nr_above, lower, upper