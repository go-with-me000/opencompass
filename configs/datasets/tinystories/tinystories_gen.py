from opencompass.datasets.tinystories import tinystoriesDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

tinystories_reader_cfg = dict(
    input_columns=['text'],
    output_column='label',
    test_split='validation')

tinystories_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: '{text}',
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

tinystories_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

tinystories_datasets = [
    dict(
        type=tinystoriesDataset,
        path='roneneldan/TinyStories',
        reader_cfg=tinystories_reader_cfg,
        infer_cfg=tinystories_infer_cfg,
        eval_cfg=tinystories_eval_cfg)
]
