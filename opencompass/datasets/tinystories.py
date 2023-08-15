from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


# Create a tinystoriesDataset class
@LOAD_DATASET.register_module()
class tinystoriesDataset(BaseDataset):

    # Load the dataset
    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        # Define a preprocess function
        def preprocess(example):
            # Set the label to 0
            example['label'] = 0
            return example

        # Map the preprocess function to the dataset
        dataset = dataset.map(preprocess)
        return dataset