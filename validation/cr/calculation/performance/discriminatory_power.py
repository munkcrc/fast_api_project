from typing import Dict, Sequence, Tuple, TypeVar, Union
from scipy.stats import norm

import numpy as np

T = TypeVar('T')
Vector = Union[Sequence[T], np.ndarray]


def gini(predictions: Vector[float],
         outcomes: Vector[int]) -> Tuple[float, Dict]:
    """
        Calculate the Gini (accuracy_ratio) for a CAP Curve
    Args:
       predictions: a vector of predictions
       outcomes: a vector of observed outcomes
    Returns:
       a tuple with:
            the gini value,
            a dictionary with intermediate results:
                the x axis for the CAP Curves,
                the y axis for the current CAP curve and perfect CAP curve,
    """
    if len(predictions) != len(outcomes):
        raise ValueError('start and end are not  of same length'
                         f"\n{' '*len('ValueError:')} "
                         f"len(start)={len(predictions)}, len(end)={len(outcomes)}")

    data = np.array([predictions, outcomes]).transpose()

    # sort data in descending order. First sort by target variable, then prediction.
    ind = np.lexsort((-data[:, 1], -data[:, 0]))
    data_sorted = data[ind]

    target_sorted = data_sorted[:, 1].astype(int)
    nr_of_true_targets = np.sum(target_sorted, dtype=int)
    # append 0 to get a (0, 0) coordinate
    y_axis_model = np.append(0, np.cumsum(target_sorted) / nr_of_true_targets)

    y_axis_perfect = np.ones(y_axis_model.shape[0])
    y_axis_perfect[0:nr_of_true_targets] = np.arange(nr_of_true_targets) / nr_of_true_targets

    x_axis = np.linspace(0, 1, y_axis_model.shape[0])

    # accuracy_ratio
    # to calculate areas we use https://en.wikipedia.org/wiki/Trapezoidal_rule

    dx = 1 / y_axis_model.shape[0]

    # notice that y_axis[0] = 0.0 and y_axis[-1] = 1.0 for both y_axis
    area_model = np.sum(y_axis_model[1:-1]) * dx + 0.5 * dx - 0.5
    area_perfect = np.sum(y_axis_perfect[1:-1]) * dx + 0.5 * dx - 0.5
    gini_value = area_model / area_perfect

    dict_intermediate = {
        'x_axis': x_axis,
        'y_axis_model': y_axis_model,
        'y_axis_perfect': y_axis_perfect
    }

    return gini_value, dict_intermediate


def auc(ratings: Vector[float],
        outcomes: Vector[int]) -> Tuple[float, float]:
    """
        Calculate the auc (area under the curve) for a ROC Curve.
        See https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    Args:
       ratings: a vector of ratings
       outcomes: a vector of observed outcomes
    Returns:
       a tuple with:
            the auc value,
            the estimated standard deviation of the auc value
    """
    if len(ratings) != len(outcomes):
        raise ValueError('start and end are not  of same length'
                         f"\n{' '*len('ValueError:')} "
                         f"len(start)={len(ratings)}, len(end)={len(outcomes)}")
    mask_true = outcomes == 1
    ratings_true = ratings[mask_true]
    ratings_false = ratings[~mask_true]
    n_true = ratings_true.size
    n_false = ratings_false.size

    # vector of length n_true. Notice we have not divided with n_false yet.
    v_10 = (np.sum(ratings_true[:, None] < ratings_false, axis=1) +
            0.5 * np.sum(ratings_true[:, None] == ratings_false, axis=1))

    # vector of length n_false. Notice we have not divided with n_true yet.
    v_01 = (np.sum(ratings_true < ratings_false[:, None], axis=1) +
            0.5 * np.sum(ratings_true == ratings_false[:, None], axis=1))

    # s = estimated standard deviation of auc
    # notice: np.var(x, ddof=1) is unbiased sample variance of vector x
    s = np.sqrt(
        np.var(v_10 / n_false, ddof=1) / n_true +
        np.var(v_01 / n_true, ddof=1) / n_false
    )
    # u = Mann-Whitney U statistic
    u = np.sum(v_10)

    auc_value = u / (n_true*n_false)

    return auc_value, s


def auc_benchmark_test(
        auc_initial,
        auc_current,
        std_dev_current,
):
    """
    The current discriminatory power is benchmarked against the discriminatory power
    measured at the time of the initial validation in the course of the model’s
    development.
    The AUC for the relevant observation period is compared with the AUC at the time of
    the initial validation during development via hypothesis testing based on a normal
    approximation, assuming a deterministic AUC at the time of development. The null
    hypothesis of the test is that the AUC at the time of development is smaller than
    the AUC for the relevant observation period.
    H0: initial AUC ≤  current AUC
    (it is lower/left-tailed Z-test BUT calculated as af right tail test)

    """
    # test statistic (S in IRB document)
    # notice that in IRB document,
    # S is calculated so it looks like a upper/right-tailed Z-test
    test_statistic = (auc_initial-auc_current)/std_dev_current
    p_value = 1 - norm.cdf(test_statistic, loc=0, scale=1)

    return test_statistic, p_value


