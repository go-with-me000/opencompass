from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import general_postprocess

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class EMEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [
            general_postprocess(prediction) for prediction in predictions
        ]
        processed_answers = [[general_postprocess(j) for j in i]
                             for i in references]

        cnt = 0
        outputs = []
        for pred, ans, origin_ans in zip(predictions, processed_answers,
                                         references):
            answers = list(set(ans + origin_ans))
            output = {'pred': pred, 'answers': answers}
            if pred in ans or pred in origin_ans:
                cnt += 1
                output['right'] = True
            else:
                output['right'] = False
            outputs.append(output)

        score = cnt / len(predictions) * 100

        return {'score': score, 'outputs': outputs}
