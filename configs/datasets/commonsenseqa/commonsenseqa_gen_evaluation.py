from opencompass.openicl import ZeroRetriever, RougeEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import MDLRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import commonsenseqaDataset, commonsenseqa_postprocess
from opencompass.utils.text_postprocessors import first_capital_postprocess

commonsenseqa_reader_cfg = dict(
    input_columns=["question", "A", "B", "C", "D", "E"],
    output_column="answer",
    test_split="validation")

_ice_template = dict(
    type=PromptTemplate,
    template="Q:\n{question}\nA: "
)

commonsenseqa_infer_cfg = dict(
    prompt_template=_ice_template,
    retriever=dict(
        type=ZeroRetriever,
    ),
    inferencer=dict(type=GenInferencer,max_out_len=50),
)

commonsenseqa_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_postprocessor=dict(type=commonsenseqa_postprocess),
)

commonsenseqa_datasets = [
    dict(
        type=commonsenseqaDataset,
        path="commonsense_qa",
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg,
        eval_cfg=commonsenseqa_eval_cfg,
    )
]

del _ice_template
