from opencompass.openicl import RougeEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import piqaDataset_V2, piqa_postprocess, piqaDataset_V3
from opencompass.utils.text_postprocessors import first_capital_postprocess

piqa_reader_cfg = dict(
    input_columns=["goal", "sol1", "sol2"],
    output_column="answer",
    test_split="validation")

piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{goal} "
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,max_out_len=50),
)

piqa_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_postprocessor=dict(type=piqa_postprocess),
)

piqa_datasets = [
    dict(
        abbr="piqa",
        type=piqaDataset_V3,
        path="piqa",
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
