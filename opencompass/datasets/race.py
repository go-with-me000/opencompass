from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class RaceDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = load_dataset(path, name)

        def preprocess(x):
            for ans, option in zip(['A', 'B', 'C', 'D'], x['options']):
                x[ans] = option
            del x['options']
            return x

        return dataset.map(preprocess)


@LOAD_DATASET.register_module()
class RaceDataset_V2(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = load_dataset(path, name)

        def preprocess(x):
            for ans, option in zip(['A', 'B', 'C', 'D'], x['options']):
                x[ans] = option
            del x['options']
            x['output'] = x[x['answer']]
            return x

        return dataset.map(preprocess)


@TEXT_POSTPROCESSORS.register_module()
def race_postprocess(text: str) -> str:
    text = text.strip()
    text = text.split('\n')[0]
    return text
