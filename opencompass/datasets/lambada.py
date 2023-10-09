import re
import string

from datasets import DatasetDict, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset
from .huggingface import get_local_datasets


@LOAD_DATASET.register_module()
class lambadaDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs, split='test')

        # dataset = get_local_datasets(split='test', **kwargs)

        def preprocess(example):
            prompt, target = example['text'].strip().rsplit(' ', 1)
            example['prompt'] = prompt
            example['label'] = target
            return example

        dataset = dataset.map(preprocess)
        return DatasetDict({'test': dataset})


@ICL_EVALUATORS.register_module()
class LambadaEvaluator(BaseEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        score = 0.0
        outputs = []
        for pred, refer in zip(predictions, references):

            pred = pred.strip().split(' ')[0]
            pred = re.split(f'[{string.punctuation}]', pred)[0]
            pred = general_postprocess(pred)
            refer = general_postprocess(refer)
            output = {'pred': pred, 'answers': refer}
            if pred == refer:
                output['right'] = True
            else:
                output['right'] = False
            outputs.append(output)
            score += pred == refer
        score = 100.0 * score / len(predictions)
        return dict(accuracy=score, outputs=outputs)
