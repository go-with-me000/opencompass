from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    dict(
        type=InternLM,
        path="/cpfs01/shared/alillm2/alillm2_hdd/zhangshuo/ckpt/0831/20B_newton_stage2/101000/",
        tokenizer_path='/cpfs01/shared/public/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path="/cpfs01/user/chenkeyu1/train_internlm/",
        model_config="/cpfs01/user/chenkeyu1/train_internlm/configs/20B_newton_stage2.py",
        model_type="LLAMA",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),

    # dict(
    #     abbr="llama-7b-gsm8k",
    #     type=HuggingFaceCausalLM,
    #     path="/cpfs01/user/chenkeyu1/model/llama-7b-hf/",
    #     tokenizer_path="/cpfs01/user/chenkeyu1/model/tokenizer/",
    #     peft_path="/cpfs01/user/chenkeyu1/program/llm/lora/07_18/epoch_9/adapter.bin",
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
]
