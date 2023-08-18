import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    mapping = {
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4]
                    }
                    target_value = mapping.get(row[5])
                    assert len(row) == 6
                    raw_data.append({
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'target': row[5],
                        'target_value': target_value,
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


@LOAD_DATASET.register_module()
class MMLU2Dataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    mapping = {
                        'A': 1,
                        'B': 2,
                        'C': 3,
                        'D': 4
                    }
                    target_value = mapping.get(row[5])
                    assert len(row) == 6
                    raw_data.append({
                        'input': row[0],
                        '1.': row[1],
                        '2.': row[2],
                        '3.': row[3],
                        '4.': row[4],
                        'target': row[5],
                        'target_value': target_value,
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset
