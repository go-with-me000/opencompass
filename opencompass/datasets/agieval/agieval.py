import json
import os.path as osp

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset
from .math_equivalence import is_equiv
from .post_process import parse_math_answer


@LOAD_DATASET.register_module()
class AGIEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str):
        from .dataset_loader import load_dataset, load_dataset_as_result_schema

        assert setting_name in 'zero-shot', 'only support zero-shot setting'
        dataset_wo_label = load_dataset(name, setting_name, path)
        dataset_with_label = load_dataset_as_result_schema(name, path)
        dataset = []
        for d1, d2 in zip(dataset_wo_label, dataset_with_label):
            dataset.append({
                'id': d2.index,
                'problem_input': d1['context'],
                'label': d2.label,
            })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class AGIEvalDataset_v2(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str):
        assert setting_name in 'zero-shot', 'only support zero-shot setting'
        filename = osp.join(path, name + '.jsonl')
        with open(filename, encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        dataset = []
        for item in data:
            passage = item['passage'] if item['passage'] else ''
            question = passage + item['question']
            options = '\n'.join(item['options']) if item['options'] else ''
            if item['label']:
                if isinstance(item['label'], list):
                    label = ''.join(item['label'])
                else:
                    label = item['label']
            else:
                label = item['answer']
            d = {'question': question, 'options': options, 'label': label}
            dataset.append(d)
        dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class AGIEvalEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        predictions = [parse_math_answer('', pred) for pred in predictions]
        outputs = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            output = {'pred': pred, 'answers': ref, 'right': False}
            if is_equiv(pred, ref):
                cnt += 1
                output['right'] = True
            outputs.append(output)
        score = cnt / len(predictions) * 100
        return {'score': score, 'outputs': outputs}


@ICL_EVALUATORS.register_module()
class AGIEvalEvaluator_v2(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        outputs = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            output = {'pred': pred, 'answers': ref, 'right': False}
            if pred == ref:
                cnt += 1
                output['right'] = True
            outputs.append(output)

        score = cnt / len(predictions) * 100

        return {'score': score, 'outputs': outputs}
