from opencompass.models import HuggingFaceCausalLM
from opencompass.models.internal import InternLM

models = [
    # dict(
    #     abbr="qwen2-7b_0",
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/share_data/feizhaoye/huggingface/Qwen/Qwen-7B_9_25/",
    #     # tokenizer_path='chiayewken/Qwen-7B',
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False,
    #                           # revision='39fc5fdcb95c8c367bbdb3bfc0db71d96266de09'
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True,
    #                       #
    #                       # revision='39fc5fdcb95c8c367bbdb3bfc0db71d96266de09'
    #                       ),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     pad_token_id=0,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

    # dict(
    #     abbr="moss_base_7b",
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/chenkeyu1/models/huggingface/moss/moss-base-7b/",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False, ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     batch_padding=True,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

    # dict(
    #     abbr="MAmmoTH-7B_0",
    #     type=HuggingFaceCausalLM,
    #     path="TIGER-Lab/MAmmoTH-7B",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False, max_length=2048),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     # batch_padding=True,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
    #
    # dict(
    #     abbr="GAIRMath_Abel-7b_0",
    #     type=HuggingFaceCausalLM,
    #     path="GAIR/GAIRMath-Abel-7b",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False, max_length=2048),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto',trust_remote_code=True),
    #     # batch_padding=True,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

    # dict(
    #     abbr=f"further_llama_7B_math_yf_3000",
    #     type=InternLM,
    #     path=f"s3://checkpoints_ssd_02/further_llama_7B_math_yf_0925/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_yf_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # )

    # dict(
    #     type=HuggingFaceCausalLM,
    #     abbr='phi-1.5-1.3b-hf',
    #     path='/mnt/petrelfs/share_data/huggingface_models/phi-1.5/phi-1_5',
    #     tokenizer_kwargs=dict(
    #         padding_side='left',
    #         truncation_side='left',
    #         trust_remote_code=True,
    #     ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     model_kwargs=dict(
    #         trust_remote_code=True,
    #         torch_dtype=None,
    #     ),
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # )

    # dict(
    #     abbr="Mistral-7B_0",
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/share_data/chenkeyu1/models/huggingface/Mistral-7B-v0.1",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           # use_fast=False,
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto',
    #                       trust_remote_code=True,
    #                       ),
    #     batch_padding=True,
    #     pad_token_id=0,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

dict(
        abbr="CodeLlama-7b-hf",
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/share_data/llm_llama/codellama_hf/CodeLlama-7b-hf",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              # use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto',
                          trust_remote_code=True,
                          ),
        batch_padding=True,
        pad_token_id=0,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
