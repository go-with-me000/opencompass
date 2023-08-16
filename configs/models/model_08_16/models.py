from opencompass.models import HuggingFaceCausalLM, InternLM

models = [
    dict(
        type=InternLM,
        path="s3://model_weights/0831/feizhaoye/3B_baijuyi_v2/12000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/my_model/model_config/config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/evaluation/open_llama_3b",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
