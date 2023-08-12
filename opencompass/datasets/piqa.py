from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class piqaDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            assert isinstance(example['label'], int)
            if example['label'] < 0:
                example['answer'] = 'NULL'
            else:
                example['answer'] = 'AB'[example['label']]
            example.pop('label')
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class piqaDataset_V3(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            assert isinstance(example['label'], int)
            if example['label'] < 0:
                example['answer'] = ''
            else:
                # example['answer'] = 'AB'[example['label']]
                example['answer'] = example['sol1'] if example[
                    'label'] == 0 else example['sol2']
            example.pop('label')
            return example

        dataset = dataset.map(preprocess)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def piqa_postprocess(text: str) -> str:
    text = text.strip()
    text = text.split('\n')[0]
    return text
