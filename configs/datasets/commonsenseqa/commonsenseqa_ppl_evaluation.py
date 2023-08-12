from opencompass.openicl import ZeroRetriever
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import commonsenseqaDataset, EvaluationEvaluator

_ice_template = dict(
    type=PromptTemplate,
    template={
        'A': "</E>Q:\n{question}\nA: {A}",
        'B': "</E>Q:\n{question}\nA: {B}",
        'C': "</E>Q:\n{question}\nA: {C}",
        'D': "</E>Q:\n{question}\nA: {D}",
        'E': "</E>Q:\n{question}\nA: {E}",
    },
    ice_token='</E>')

commonsenseqa_infer_cfg = dict(
    ice_template=_ice_template,
    retriever=dict(
        type=ZeroRetriever,
        ),
    inferencer=dict(type=PPLInferencer))

commonsenseqa_eval_cfg = dict(evaluator=dict(type=EvaluationEvaluator))

commonsenseqa_datasets = [
    dict(
        type=commonsenseqaDataset,
        path='commonsense_qa',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
            output_column='answerKey',
            test_split='validation',
        ),
        infer_cfg=commonsenseqa_infer_cfg,
        eval_cfg=commonsenseqa_eval_cfg)
]

del _ice_template
