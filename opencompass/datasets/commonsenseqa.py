from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset
from .huggingface import get_local_datasets


@LOAD_DATASET.register_module()
class commonsenseqaDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        # dataset = get_local_datasets(**kwargs)

        def pre_process(example):
            for i in range(5):
                example[chr(ord('A') + i)] = example['choices']['text'][i]
            return example

        dataset = dataset.map(pre_process).remove_columns(
            ['question_concept', 'id', 'choices'])
        return dataset
