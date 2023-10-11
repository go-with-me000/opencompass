from opencompass.datasets.tinystories import tinystoriesDataset
from opencompass.openicl import LMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

tinystories_reader_cfg = dict(
    input_columns=['text','prompt'],
    output_column='answer',
    test_range="[0:10]",
    test_split='validation')

tinystories_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

tinystories_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role="SYSTEM",
                            fallback_role="HUMAN",
                            prompt="Please rate the fluency of the following sentence on a scale of 1-10, with 1 being the lowest and 10 being the highest. No need to provide any other information except the number."
                        ),
                    ],
                    round=[dict(role="HUMAN",
                                prompt="{prompt} {prediction}")]))),
        pred_role="BOT",
    )

tinystories_datasets = [
    dict(
        type=tinystoriesDataset,
        path='roneneldan/TinyStories',
        reader_cfg=tinystories_reader_cfg,
        infer_cfg=tinystories_infer_cfg,
        eval_cfg=tinystories_eval_cfg)
]
