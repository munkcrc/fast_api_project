from data_controller.load_data import load_data
from model.dataset import Dataset


class DqService:

    def __init__(self):
        self.dataset = Dataset(dataset=load_data())

    @property
    def dataset(self):
        return self.dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

