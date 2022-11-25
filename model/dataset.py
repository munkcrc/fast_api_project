from pydantic import BaseModel
from model.segmentaion import Segmentation


class DatasetModel(BaseModel):
    id: str
    root_data: dict
    observations: str
    segmentations: list[Segmentation] = []
