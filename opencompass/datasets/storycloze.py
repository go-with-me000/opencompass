from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset
from .huggingface import get_local_datasets


@LOAD_DATASET.register_module()
class storyclozeDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        # special process
        dataset = load_dataset(**kwargs, split='train+eval')

        # dataset = get_local_datasets(split='train+eval', **kwargs)

        def preprocess(example):
            example['context'] = ' '.join([
                example['input_sentence_1'], example['input_sentence_2'],
                example['input_sentence_3'], example['input_sentence_4']
            ])
            return example

        dataset = dataset.map(preprocess)

        return DatasetDict({'test': dataset})


@LOAD_DATASET.register_module()
class storyclozeDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        # special process
        dataset = load_dataset(**kwargs, split='train+eval')

        def preprocess(example):
            example['context'] = ' '.join([
                example['input_sentence_1'], example['input_sentence_2'],
                example['input_sentence_3'], example['input_sentence_4']
            ])
            example['answer_right_ending'] = ' AB'[
                example['answer_right_ending']]
            return example

        dataset = dataset.map(preprocess)
        return dataset
