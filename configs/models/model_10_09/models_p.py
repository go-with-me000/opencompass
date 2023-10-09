from opencompass.models.internal import InternLM

models = [
    # dict(
    #     abbr="1b_v6_28000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v6/28000",
    #     tokenizer_path='/mnt/petrelfs/share_data/wangbo/wiki_9998_small.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1B_v6.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="1b_v6_32000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v6/32000",
    #     tokenizer_path='/mnt/petrelfs/share_data/wangbo/wiki_9998_small.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1B_v6.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
]

# for item in range(0, 90):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v12_random_{100 + item*100}",
#         type=InternLM,
#         path=f"/mnt/petrelfs/share/songdemin/model/7B_llama2_coder_sft_v12_random/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v12_random.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)


for item in range(0, 18):
    model_info = dict(
        abbr=f"7B_llama2_coder_sft_v15_{500 + item*500}",
        type=InternLM,
        path=f"/mnt/petrelfs/share/songdemin/model/7B_llama2_coder_sft_v15/{500 + item*500}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v15.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)