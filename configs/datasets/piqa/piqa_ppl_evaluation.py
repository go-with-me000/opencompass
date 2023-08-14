from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset, EvaluationEvaluator

piqa_reader_cfg = dict(
    input_columns=['goal', 'sol1', 'sol2'],
    output_column='label',
    test_split='validation')

piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: '{goal} {sol1}\n',
            1: '{goal} {sol2}\n'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

piqa_eval_cfg = dict(evaluator=dict(type=EvaluationEvaluator))

piqa_datasets = [
    dict(
        type=HFDataset,
        path='piqa',
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
