from opencompass.openicl import RougeEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import RaceDataset, race_postprocess, RaceDataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='output')

race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=
        '{article}\n\nQ: {question}\nA:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,max_out_len=50))

race_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_postprocessor=dict(type=race_postprocess))

race_datasets = [
    dict(
        type=RaceDataset_V2,
        abbr='race-middle',
        path='race',
        name='middle',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg),
    dict(
        type=RaceDataset_V2,
        abbr='race-high',
        path='race',
        name='high',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg)
]
