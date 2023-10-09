import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SubjectivityDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    raw_data.append({'input': row[0], 'output': row[3]})
            dataset[split] = Dataset.from_list(raw_data)
        return dataset
