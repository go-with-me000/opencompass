# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models import LLama, HuggingFaceCausalLM, InternLM

models = [
    # dict(
    #     abbr="llama-7b",
    #     type=HuggingFaceCausalLM,
    #     path="/fs-computility/llm/chenkeyu1/models/llama-7b-hf",
    #     # tokenizer_path='Qwen/Qwen-7B',
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False,
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True,
    #                       ),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

    dict(
        type=InternLM,
        abbr="7B_dufu-coder_v1_14000",
        path="/fs-computility/llm/shared/llm_data/ckpts/7B_dufu-coder_v1/14000",
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llamav4.model',
        tokenizer_type='v4',
        module_path="/fs-computility/llm/shared/gaoyang/code/train_internlm/",
        model_config="/fs-computility/llm/chenkeyu1/train_internlm/configs/7B_dufu-coder_v1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]
