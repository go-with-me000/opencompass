import json
import os
from datasets import Dataset
from datasets import load_dataset, load_from_disk
import os.path as osp
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

# Create a tinystoriesDataset class
@LOAD_DATASET.register_module()
class StoryDataset(BaseDataset):

    # Load the dataset
    @staticmethod
    def load(path, name):
        path = osp.join(path, f"{name}.json")

        with open(path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        output_list = data.get("output", [])
        prompts = []
        answers = []
        for item in output_list:
            prompt = item.get("prompt", "")
            prompts.append(prompt)
            answer = item.get("answer", "")
            answers.append(answer)
        dataset = Dataset.from_dict({
            'prompt': prompts,
            'answer': answers
        })
        return dataset