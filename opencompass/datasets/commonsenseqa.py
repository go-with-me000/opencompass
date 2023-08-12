from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class commonsenseqaDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def pre_process(example):
            for i in range(5):
                example[chr(ord('A') + i)] = example['choices']['text'][i]
            if example['answerKey']:
                example['answer'] = example[example['answerKey']]
            else:
                example['answer'] = 'NULL'
            return example

        def filter_fun(example):
            return example['answer'] != 'NULL'

        dataset = dataset.map(pre_process).remove_columns(
            ['question_concept', 'id', 'choices'])
        dataset = dataset.filter(filter_fun)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def commonsenseqa_postprocess(text: str) -> str:
    text = text.strip()
    text = text.split('\n')[0]
    return text
