from opencompass.openicl import BaseEvaluator
from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('gsm8k_dataset')
def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')


@TEXT_POSTPROCESSORS.register_module('gsm8k')
def gsm8k_postprocess(text: str) -> str:
    text = text.split('\n\n')[0]
    text = text.split(' ')[::-1]
    flag = False
    ret = ''
    for i in range(len(text)):
        s = text[i]
        for i in range(len(s)):
            if s[i].isdigit():
                flag = True
                ret = s
                break
        if flag:
            break
    ret1 = ''
    for i in range(len(ret)):
        if ret[i].isdigit():
            ret1 += ret[i]
    return ret1


class gsm8kEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        outputs = []
        for i, j in zip(predictions, references):
            output = {'pred': i, 'answers': j, 'right': False}
            count += 1
            if i == j:
                correct += 1
                output['right'] = True
            outputs.append(output)
        result = {'accuracy': 100 * correct / count, 'outputs': outputs}
        return result
