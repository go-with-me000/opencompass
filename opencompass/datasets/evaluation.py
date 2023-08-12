import random
import statistics
from typing import List

import evaluate
import numpy as np
from datasets import load_dataset

from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@ICL_EVALUATORS.register_module()
class EvaluationEvaluator():

    def __init__(self) -> None:
        self.metric = 'accuracy'
        self.seed = 0

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        mapping_to_int_dict = {
            label: idx
            for idx, label in enumerate(set(map(str, references)))
        }
        pred_set = set(predictions)
        for pred in pred_set:
            if str(pred) not in mapping_to_int_dict.keys():
                mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
        golds = [mapping_to_int_dict[str(gold)] for gold in references]
        preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
        return {'predictions': preds, 'references': golds}

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores['accuracy'] *= 100
        scores['std'] *= 10000
        return scores

    def score(self, pred_dicts: List, references: List) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        predictions, prompts, meanP, meanB, meanV, meanRP, meanRB = self._calculate_pred(
            pred_dicts)
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }
        path = '/mnt/petrelfs/chenkeyu1/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--accuracy/' \
               'f887c0aab52c2d38e1f8a215681126379eca617f96c447638f751434e8e65b14/accuracy.py'
        self.metric = path
        metric = evaluate.load(self.metric)
        result_dict = self._preprocess(predictions, references)
        predictions, references = result_dict['predictions'], result_dict[
            'references']
        scores = metric.compute(predictions=predictions, references=references)

        random.setstate(random_state)
        np.random.set_state(np_random_state)
        outputs = {}
        for i, (pred, refer,
                prompt) in enumerate(zip(predictions, references, prompts)):
            outputs[i] = {
                'prompt': prompt,
                'predictions': pred,
                'reference': refer,
                'result': 'yes' if pred == refer else 'no'
            }
        scores['std'] = meanV
        scores['ppl'] = meanP
        scores['bpb'] = meanB
        scores['right-ppl'] = meanRP
        scores['right-bpb'] = meanRB
        scores['outputs'] = outputs
        result = self._postprocess(scores)
        return result

    def _calculate_pred(self, pred_dicts: List):
        predictions = []
        prompt_list = []
        std_list = []
        ppl_list = []
        right_ppl_list = []
        right_bpb_list = []
        bpb_list = []
        for pred_dict in pred_dicts:
            preds = {
                key: value
                for key, value in pred_dict.items()
                if key.startswith('label: ')
            }
            keys = []
            values = []
            for item in preds.items():
                keys.append(item[0])
                values.append(item[1])
            keys = [key.replace('label: ', '') for key in keys]
            prompts = [value['prompt'] for value in values]
            ppls = [value['PPL'] for value in values]
            bpbs = [value['BPB'] for value in values]
            pred = keys[ppls.index(min(ppls))]
            prompt = prompts[ppls.index(min(ppls))]
            stds = statistics.stdev(ppls)
            std_list.append(stds)
            ppl_list.append(statistics.mean(ppls))
            right_ppl_list.append(min(ppls))
            bpb_list.append(statistics.mean(bpbs))
            right_bpb_list.append(min(bpbs))
            predictions.append(pred)
            prompt_list.append(prompt)
        meanP = statistics.mean(ppl_list)
        meanB = statistics.mean(bpb_list)
        meanV = statistics.mean(std_list)
        meanRP = statistics.mean(right_ppl_list)
        meanRB = statistics.mean(right_bpb_list)
        return predictions, prompt_list, meanP, meanB, meanV, meanRP, meanRB
