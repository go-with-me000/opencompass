from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    from .summarizers.example import summarizer
    from .lark import lark_bot_url
    from .datasets.collections.example import datasets

models = [
    # LLaMA 7B(Inference without padding)
    dict(
        type=HuggingFaceCausalLM,
        # path='weight:s3://model_weights/hf/65B/',
        # path="decapoda-research/llama-7b-hf",
        path="decapoda-research/llama-7b-hf",
        tokenizer_path='decapoda-research/llama-7b-hf',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              proxies={
                                  'http': 'http://10.1.8.5:32680',
                                  'https': 'http://10.1.8.5:32680',
                              }),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
