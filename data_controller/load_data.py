import numpy as np
import pandas as pd
from validation.cr.data import DataSet
from model.dataset import DatasetModel


def load_data():
    df_in = pd.read_parquet(
        "C:/Users/JonasChristensen/Workspace/fast_api_dojo/fast_api_project/data_controller/presentation.parquet")
    str_cols = df_in.select_dtypes(include=['object']).columns
    df_in[str_cols] = df_in[str_cols].applymap(lambda x: np.unicode_(x))

    dataset = DataSet('data', df_in.replace(np.nan, 'nan', regex=True).head(5))

    return DatasetModel(id=dataset.id,
                        root_data=dataset._df.to_dict(orient='index'),
                        observations=dataset.observations,
                        segmentations=dataset.segmentations)
