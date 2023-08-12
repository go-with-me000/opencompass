from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class hellaswagDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            for i in range(4):
                example[chr(ord('A') + i)] = example['endings'][i]
            return example

        dataset = dataset.map(preprocess).remove_columns(['endings'])
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            for i in range(4):
                example[chr(ord('A') + i)] = example['endings'][i]
            if example['label']:
                example['label'] = 'ABCD'[int(example['label'])]
            else:
                example['label'] = 'NULL'
            return example

        dataset = dataset.map(preprocess).remove_columns(['endings'])
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V3(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            for i in range(4):
                example[chr(ord('A') + i)] = example['endings'][i]
            if example['label']:
                example['label'] = 'ABCD'[int(example['label'])]
                example['answer'] = example[example['label']]
            else:
                example['label'] = 'NULL'
                example['answer'] = ''
            return example

        def filter_fun(example):
            return example['label'] != 'NULL'

        dataset = dataset.map(preprocess).remove_columns(['endings'])
        dataset = dataset.filter(filter_fun)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def hellaswag_postprocess(text: str) -> str:
    text = text.strip()
    text = text.split('\n')[0]
    return text
