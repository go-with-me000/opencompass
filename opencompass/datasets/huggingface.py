import os

from datasets import concatenate_datasets, load_dataset, load_from_disk

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


def get_local_datasets(split=None, **kwargs):
    root_path = None
    if os.path.exists('/cpfs01'):
        root_path = '/cpfs01/shared/public/chenkeyu1/datasets/data/'
    elif os.path.exists('/fs-computility'):
        root_path = '/fs-computility/llm/shared/chenkeyu1/datasets/'
    if root_path is not None:
        path = kwargs.get('path')
        name = kwargs.get('name', None)
        data_files = kwargs.get('data_files', None)
        if data_files is not None:
            return load_dataset(**kwargs)
        if name is not None:
            route = root_path + path + '/' + name
        else:
            route = root_path + path + '/'
        if split == 'test':
            route = route + '/test'
        dataset = load_from_disk(route)
        if split == 'train+eval':

            dataset = concatenate_datasets([dataset['train'], dataset['eval']])
        return dataset
    else:
        if split != None:
            return load_dataset(**kwargs, split=split)
        return load_dataset(**kwargs)


@LOAD_DATASET.register_module()
class HFDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        # return load_dataset(**kwargs)
        return get_local_datasets(**kwargs)
