from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


class HFDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        return load_dataset(**kwargs)
