from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import RaceDataset, EvaluationEvaluator

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer')

race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A':
            '{article}\n\nQ: {question}\nA: {A}',
            'B':
            '{article}\n\nQ: {question}\nA: {B}',
            'C':
            '{article}\n\nQ: {question}\nA: {C}',
            'D':
            '{article}\n\nQ: {question}\nA: {D}',
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

race_eval_cfg = dict(evaluator=dict(type=EvaluationEvaluator))

race_datasets = [
    dict(
        type=RaceDataset,
        abbr='race-middle',
        path='race',
        name='middle',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg),
    dict(
        type=RaceDataset,
        abbr='race-high',
        path='race',
        name='high',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg)
]
