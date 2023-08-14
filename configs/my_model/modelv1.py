from opencompass.models import HuggingFaceCausalLM, InternLM

models = [
    # dict(
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/feizhaoye/0803/model_ckpt/7B_dufu/5000/",
    #     # tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     # tokenizer_type='v4',
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/my_model/model_config/config.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)),
    #
    # dict(
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/feizhaoye/0803/model_ckpt/7B_dufu/10000/",
    #     # tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     # tokenizer_type='v4',
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/my_model/model_config/config.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1))

# dict(
#         type=InternLM,
#         path="s3://checkpoints_ssd_02/feizhaoye/0803/model_ckpt/7B_dufu/5000/",
#         # tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
#         # tokenizer_type='v4',
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
#         tokenizer_type='v7',
#         model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/my_model/model_config/config.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)),

    dict(
        type=InternLM,
        path="s3://model_weights/0831/feizhaoye/7B_dufu/33000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/my_model/model_config/config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))

# dict(
#         type=HuggingFaceCausalLM,
#         abbr='Qwen',
#         path="/mnt/petrelfs/chenkeyu1/models/huggingface/Qwen-7B/",
#         tokenizer_path='Qwen/Qwen-7B',
#         tokenizer_kwargs=dict(padding_side='left',
#                               truncation_side='left',
#                               use_fast=False,
#                               trust_remote_code=True,
#                               ),
#         max_out_len=100,
#         max_seq_len=18000,
#         batch_size=1,
#         model_kwargs=dict(device_map='auto',trust_remote_code=True),
#         batch_padding=False,  # if false, inference with for-loop without batch padding
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     ),
]
