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
        dataset = load_dataset(**kwargs)
        # dataset = get_local_datasets(**kwargs)

        # Define a preprocess function
        def preprocess(example):
            # Set the label to 0
            example['label'] = 0
            text = example['text']
            last_space_index = text[:50].rfind(' ')

            if last_space_index != -1:
                # 如果找到了空格，截取到最后一个空格之前的部分
                example["prompt"] = text[:last_space_index]
                example["answer"] = text[last_space_index + 1:]

            else:
                example["prompt"] = text[:50]
                example["answer"] = text[50:]
            return example

        def filter_function(example):
            return len(example['text']) >= 100
        # dataset = dataset.filter(filter_function)
        # Map the preprocess function to the dataset
        dataset = dataset.filter(filter_function).map(preprocess)
        return dataset