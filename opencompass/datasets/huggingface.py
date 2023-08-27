import os

from datasets import load_dataset, load_from_disk

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HFDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        if os.path.exists("/cpfs01"):
            path = kwargs.get("path")
            name = kwargs.get("name", None)
            data_files = kwargs.get("data_files", None)
            if data_files is not None:
                return load_dataset(**kwargs)
            if name is not None:
                route = "/cpfs01/shared/public/chenkeyu1/datasets/data/" + path + "/" + name
            else:
                route = "/cpfs01/shared/public/chenkeyu1/datasets/data/" + path + "/"
            dataset = load_from_disk(route)
            return dataset
        else:
            return load_dataset(**kwargs)
