import pandas as pd
import numpy as np


def get_data_extend(
             df: pd.DataFrame,
             rating_group_name: str,
             observations_name: str,
             defaulted_name: str,
             pd_name: str,):

    df[f'{pd_name} min'] = df[pd_name]
    df[f'{pd_name} max'] = df[pd_name]

    df_out = (df.groupby(rating_group_name, as_index=False).agg(
        {observations_name: 'count',
         defaulted_name: 'sum',
         f'{pd_name} min': 'min',
         pd_name: 'mean',
         f'{pd_name} max': 'max'}).rename(
        columns={rating_group_name: 'Rating Group',
                 observations_name: 'Observations',
                 defaulted_name: 'Defaulted',
                 f'{pd_name} min': 'PD Min',
                 pd_name: 'PD',
                 f'{pd_name} max': 'PD Max'})).copy()

    data_out = (list(zip(
        df_out['Rating Group'].values,
        df_out['Observations'].values,
        df_out['Defaulted'].values,
        df_out['PD'].values)))

    return df_out, data_out


def get_data(df: pd.DataFrame,
             rating_group_name: str,
             observations_name: str,
             defaulted_name: str,
             pd_name: str,):

    df_out = (df.groupby(rating_group_name, as_index=False).agg(
        {observations_name: 'count', defaulted_name: 'sum', pd_name: 'mean'}).rename(
        columns={rating_group_name: 'Rating Group',
                 observations_name: 'Observations',
                 defaulted_name: 'Defaulted',
                 pd_name: 'PD'})).copy()

    data_out = (list(zip(
        df_out['Rating Group'].values,
        df_out['Observations'].values,
        df_out['Defaulted'].values,
        df_out['PD'].values)))

    return df_out, data_out


def scoring_group_quantiles(df, pd_name, quantile_limits):
    quantile_pd_values = [np.percentile(df[pd_name], i) for i in quantile_limits]
    quantile_pd_values = [0] + quantile_pd_values + [1]
    conditions = []
    values = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWYZ'
    for i, q_value in enumerate(quantile_pd_values[:-1]):
        lower = quantile_pd_values[i]
        upper = quantile_pd_values[i + 1]
        conditions.append(
            (lower < df[pd_name]) & (df[pd_name] <= upper))
        values.append(alphabet[i])

    return quantile_pd_values, conditions, values


def scoring_group_pd_intervals(df, pd_name, quantile_pd_values):
    quantile_pd_values = [0] + quantile_pd_values + [1]
    conditions = []
    values = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWYZ'
    for i, q_value in enumerate(quantile_pd_values[:-1]):
        lower = quantile_pd_values[i]
        upper = quantile_pd_values[i + 1]
        conditions.append(
            (lower < df[pd_name]) & (df[pd_name] <= upper))
        values.append(alphabet[i])

    return quantile_pd_values, conditions, values


def scoring_group_a_score(df, pd_name):
    quantile_pd_values = [
        0,
        0.0044,
        0.0085,
        0.0195,
        0.0428,
        0.0838,
        0.0951,
        0.1101,
        0.1301,
        0.1601,
        0.2001,
        0.25,
        1
    ]
    values = [
        'A',
        'B',
        'C',
        'D',
        'E',
        'F1',
        'F2',
        'F3',
        'F4',
        'F5',
        'F6',
        'Rejected',
    ]
    conditions = []
    for i, q_value in enumerate(quantile_pd_values[:-1]):
        lower = quantile_pd_values[i]
        upper = quantile_pd_values[i + 1]
        conditions.append(
            (lower < df[pd_name]) & (df[pd_name] <= upper))

    return quantile_pd_values, conditions, values
