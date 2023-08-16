import os

from datasets import load_dataset, load_from_disk

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


# Create a tinystoriesDataset class
@LOAD_DATASET.register_module()
class tinystoriesDataset(BaseDataset):

    # Load the dataset
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
        else:
            dataset = load_dataset(**kwargs)

        # Define a preprocess function
        def preprocess(example):
            # Set the label to 0
            example['label'] = 0
            return example

        # Map the preprocess function to the dataset
        dataset = dataset.map(preprocess)
        return dataset