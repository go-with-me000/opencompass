from opencompass.datasets import QwenDataset
from opencompass.openicl import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

qwen_reader_cfg = dict(
    input_columns=['text'],
    output_column='text')

qwen_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={0:"{text}"}),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

qwen_datasets = [
    dict(
        type=QwenDataset,
        abbr='Qwen',
        path='/mnt/petrelfs/chenkeyu1/datasets/zhouyunhua/qwen/',
        reader_cfg=qwen_reader_cfg,
        infer_cfg=qwen_infer_cfg)
]
