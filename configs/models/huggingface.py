from opencompass.models import HuggingFaceCausalLM, InternLM

models = [
    # dict(
    #     abbr="qwen-7b",
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/share_data/chenkeyu1/models/huggingface/Qwen-7B/",
    #     tokenizer_path='Qwen/Qwen-7B',
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False, ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

    dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/huggingface/liuxiaoran/epoch_1/",
        tokenizer_path="/mnt/petrelfs/chenkeyu1/models/collie/tokenizer/",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
