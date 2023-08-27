import os

from datasets import load_dataset, load_from_disk

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class RaceDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        if os.path.exists("/cpfs01"):
            route = "/cpfs01/shared/public/chenkeyu1/datasets/data/" + path + "/" + name
            dataset = load_from_disk(route)
        else:
            dataset = load_dataset(path, name)

        def preprocess(x):
            for ans, option in zip(['A', 'B', 'C', 'D'], x['options']):
                x[ans] = option
            del x['options']
            return x

        return dataset.map(preprocess)
