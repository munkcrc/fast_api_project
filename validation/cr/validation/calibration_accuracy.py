# import matplotlib.pyplot as plt # overvej plotly i stedet
from typing import Callable, List, Tuple  # ,Any, Dict, Iterable,
# from functools import cache, cached_property
# from itertools import accumulate
# import pandas as pd
import scipy.stats

from pandas import DataFrame, Series


def format_to_percentage(df, column_name):
    df[column_name] = Series(["{0:.3f}%".format(val * 100) for val in df[column_name]])


def add_total_row_to_data(data: List[Tuple[str, int, int, float]]):
    observations = [b for _, b, _, _ in data]
    defaults = [c for _, _, c, _ in data]
    pds = [d for _, _, _, d in data]

    total_obs = sum(observations)
    total_defaults = sum(defaults)
    total_pd = sum(b*d for _, b, _, d in data)/total_obs

    observations += [total_obs]
    defaults += [total_defaults]
    pds += [total_pd]

    data_out = data + [('total', total_obs, total_defaults, total_pd)]

    return data_out, observations, defaults, pds


def actual_vs_model_deviation(data: List[Tuple[str, int, int, float]],
                              as_df: bool = True,):
    """
    Returns a table on format
    Rating Group, Observations, Defaulted, PD, DF, DF-PD, (DF-PD)/PD
    """
    data, observations, defaults, pds = add_total_row_to_data(data)

    dfs = [nr_def/nr_obs for nr_def, nr_obs in zip(defaults, observations)]
    pd_df_diff = [pd-df for pd, df in zip(pds, dfs)]
    df_pd_relative = [diff/df for diff, df in zip(pd_df_diff, dfs)]

    data_out = [
        t + (df,) + (diff,) + (relative,)
        for t, df, diff, relative in zip(data, dfs, pd_df_diff, df_pd_relative)
    ]
    if as_df:
        df = DataFrame(data_out,
                       columns=['Rating Group', 'Observations', 'Defaulted', 'PD',
                                'DF', 'PD-DF', '(PD-DF)/DF']
                       )
        for name in ['PD', 'DF', 'PD-DF', '(PD-DF)/DF']:
            format_to_percentage(df, name)
        return df
    else:
        return data_out


def chi2_test(data: List[Tuple[str, int, int, float]],
              significance_level: float = 0.05,
              as_df: bool = True,
              print_conclusion: bool = False):

    degree_of_freedom = len(data)

    hls = [
        ((nr_obs*pd-nr_def)**2)/(nr_obs*pd*(1-pd)) for _, nr_obs, nr_def, pd in data
    ]

    test_statistic = sum(hls)

    p_value = 1 - scipy.stats.chi2.cdf(test_statistic, degree_of_freedom)

    data, observations, defaults, pds = add_total_row_to_data(data)

    data_out = [
        t + (hl,) for t, hl in zip(data, hls + [test_statistic])
    ]

    if print_conclusion:
        if p_value > significance_level:
            print(f'Since the P-value is higher than the significance level'
                  f' ({p_value:.3f} > {significance_level})')
            print(f"the hypothesis 'H0: the PDs of all the rating groups are correctly"
                  f" estimated'")
            print(f'can not be rejected')
        else:
            print(f'Since the P-value is lower than the significance level'
                  f' ({p_value:.3f} <= {significance_level})')
            print(f"the hypothesis 'H0: the PDs of all the rating groups are correctly"
                  f" estimated'")
            print(f'is rejected')

    if as_df:
        df = DataFrame(data_out,
                       columns=['Rating Group', 'Observations', 'Defaulted', 'PD',
                                'HL']
                       )
        format_to_percentage(df, 'PD')
        df['HL'] = Series(["{0:.3f}".format(val) for val in df['HL']])

        return df

    else:
        return data_out


def binomial_test_one_sided(
        data: Tuple[int, int, float],
        significance_level: float = 0.05):
    """

    null hypothesis H0: the PD of a rating category is correct (p_value > alpha)
    alternative hypothesis H1: the PD of a rating category is underestimated

    """
    nr_obs = data[0]
    nr_defaults = data[1]
    pd = data[2]
    alpha = 1 - significance_level
    k_star = int(scipy.stats.binom.ppf(q=alpha, n=nr_obs, p=pd))
    p_value = 1-scipy.stats.binom.cdf(k=nr_defaults, n=nr_obs, p=pd)

    if p_value > significance_level:
        verdict = 'Not reject'
    else:
        verdict = 'Reject'

    return k_star, p_value, verdict


def binomial_test_one_sided_table(
        data: List[Tuple[str, int, int, float]],
        significance_level: float = 0.05,
        as_df: bool = True,):

    data, observations, defaults, pds = add_total_row_to_data(data)

    binomial_outcome = [
        binomial_test_one_sided((nr_obs, nr_def, pd), significance_level)
        for _, nr_obs, nr_def, pd in data
    ]

    data_out = [
        t + b_t
        for t, b_t in zip(data, binomial_outcome)
    ]

    if as_df:
        df = DataFrame(data_out,
                       columns=['Rating Group', 'Observations', 'Defaulted', 'PD',
                                'k*', 'p-value', 'H0'])

        for name in ['PD', 'p-value']:
            format_to_percentage(df, name)
        return df
    else:
        return data_out


def wald_interval(
        data: Tuple[int, int, float],
        significance_level: float = 0.05):
    nr_obs = data[0]
    nr_defaults = data[1]
    pd = data[2]
    df = nr_defaults/nr_obs
    alpha = 1 - 0.5*significance_level
    z = scipy.stats.norm.ppf(alpha)
    term = z*(df*(1-df)/nr_obs)**0.5
    low = df - term
    high = df + term

    if low <= pd <= high:
        verdict = 'Not reject'
    else:
        verdict = 'Reject'

    return low, high, verdict


def agresti_coull_interval(
        data: Tuple[int, int, float],
        significance_level: float = 0.05):
    return wald_interval(
        data=(data[0]+4, data[1]+2, data[2]),
        significance_level=significance_level)


def interval_table(
        data: List[Tuple[str, int, int, float]],
        significance_level: float = 0.05,
        interval_function: Callable = wald_interval,
        as_df: bool = True,):

    data, observations, defaults, pds = add_total_row_to_data(data)

    interval_outcome = [
        interval_function((nr_obs, nr_def, pd), significance_level)
        for _, nr_obs, nr_def, pd in data
    ]

    data_out = [
        t + b_t
        for t, b_t in zip(data, interval_outcome)
    ]

    if as_df:
        df = DataFrame(data_out,
                       columns=['Rating Group', 'Observations', 'Defaulted', 'PD',
                                'low', 'high', 'H0'])

        for name in ['PD', 'low', 'high']:
            format_to_percentage(df, name)
        return df
    else:
        return data_out


def wald_interval_table(
        data: List[Tuple[str, int, int, float]],
        significance_level: float = 0.05,
        as_df: bool = True,):
    return interval_table(
        data,
        significance_level,
        wald_interval,
        as_df)


def agresti_coull_interval_table(
        data: List[Tuple[str, int, int, float]],
        significance_level: float = 0.05,
        as_df: bool = True,):
    return interval_table(
        data,
        significance_level,
        agresti_coull_interval,
        as_df)


"""
# https://en.wikipedia.org/wiki/Binomial_test
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
scipy.stats.binom_test(x=308, n=1689, p=0.1779, alternative='greater')

1-scipy.stats.binom.cdf(k=308-1, n=1689, p=0.1779)

# inverse cdf
scipy.stats.binom.ppf(q=0.95, n=1689, p=0.1779)
scipy.stats.binom_test(
    x=int(scipy.stats.binom.ppf(q=0.95, n=1689, p=0.1779)),
    n=1689, p=0.1779, alternative='greater')
"""