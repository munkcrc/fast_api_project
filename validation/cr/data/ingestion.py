from typing import Union, List
from uuid import uuid4
from .dataset import DataSet
from cr.automation.recordable import recordable
import re
import pandas as pd
import numpy as np

def _guess_targets(values: List[str]) -> List[str]:
    abbreviations = ["EAD", "LGD", "CR", "LGL", "CCF", "CF", "PD", "Score", "D12", "Exposure"]
    full_names = [
        "Target", "ExposureAtDefault", "LossGivenDefault", "CureRate", "LossGivenLoss", "CreditConversionFactor",
        "ConversionFactor", "ProbabilityOfDefault", "DefaultProbability"
    ]

    potential_matches = []
    for raw_value in values:
        value = raw_value.lower()
        for abbreviation in abbreviations:
            if re.match(r"(?<![a-zA-Z])%s(?![a-zA-Z])" % abbreviation.lower(), value) \
                    or re.search(r"(?<![a-zA-Z]) %s (?![a-zA-Z])" % abbreviation.lower(), value) \
                    or r" %s " % abbreviation.lower() in value:
                potential_matches.append(raw_value)

        stripped_value = re.sub(r"[_\s]", "", value)
        for term in full_names:
            if term.lower() in stripped_value:
                potential_matches.append(raw_value)
    return potential_matches

def infer_var_categories(df):
    expected_targets = _guess_targets(list(df.columns))

    expected_targets = list(dict.fromkeys(expected_targets))  # remove duplicates and
    # keep original order
    expected_segmentors = list(df.select_dtypes(include=['datetime64']).columns) + list(df.columns[df.nunique() < 12])
    expected_segmentors = list(dict.fromkeys(expected_segmentors))
    expected_segmentors = [segmentor for segmentor in expected_segmentors if segmentor not in expected_targets]
    expected_factors = df.select_dtypes(include=['number']).columns
    expected_factors = list(dict.fromkeys(expected_factors))
    expected_factors = [factor for factor in expected_factors if factor not in expected_targets]

    return expected_factors, expected_segmentors, expected_targets

def _init_dataset(df, id_):
    # if no ID is set generate one
    if not id_:
        id_ = str(uuid4())

    # Parse object dtypes into unicode strings
    str_cols = df.select_dtypes(include=['object']).columns
    df[str_cols] = df[str_cols].applymap(lambda x: np.unicode_(x))

    return DataSet(id_, df)


@recordable
def _from(read_func, path: str, id_: str = None, ingestion_kwargs: dict = None) -> DataSet:
    if not ingestion_kwargs:
        ingestion_kwargs = {}
    df = read_func(path, **ingestion_kwargs)
    return _init_dataset(df, id_)


@recordable
def from_excel(*args, **kwargs) -> DataSet:
    return _from(pd.read_excel, *args, **kwargs)


@recordable
def from_csv(*args, **kwargs) -> DataSet:
    return _from(pd.read_csv, *args, **kwargs)


@recordable
def from_parquet(*args, **kwargs) -> DataSet:
    return _from(pd.read_parquet, *args, **kwargs)


@recordable
def from_feather(*args, **kwargs) -> DataSet:
    return _from(pd.read_feather, *args, **kwargs)

