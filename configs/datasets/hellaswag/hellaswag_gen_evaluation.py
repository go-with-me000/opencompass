from opencompass.openicl import RougeEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import hellaswagDataset_V2, hellaswag_postprocess, hellaswagDataset_V3
from opencompass.utils.text_postprocessors import first_capital_postprocess

hellaswag_reader_cfg = dict(
    input_columns=["ctx", "A", "B", "C", "D"],
    output_column="answer",
    test_split="validation")

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{ctx}\n"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,max_out_len=50),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_postprocessor=dict(type=hellaswag_postprocess),
)

hellaswag_datasets = [
    dict(
        type=hellaswagDataset_V3,
        path="hellaswag",
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
