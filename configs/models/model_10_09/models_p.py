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

    dict(
        abbr="1b_v1_4000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v1/4000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="1b_v1_12000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v1/12000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="1b_v1_20000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v1/20000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="1b_v5_32000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v5/32000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
        tokenizer_type='v4',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1B_v5.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

    dict(
        abbr="further_llama_7B_math_yf_1007_3000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/further_llama_7B_math_yf_1007/3000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/",
        model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/configs/further_llama_7B_math_yf.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="further_llama_7B_math_yf_1007_5000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/further_llama_7B_math_yf_1007/5000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/",
        model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/configs/further_llama_7B_math_yf.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="further_llama_7B_math_yf_1007_8000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/further_llama_7B_math_yf_1007/8000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/",
        model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/configs/further_llama_7B_math_yf.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),    dict(
        abbr="further_llama_7B_math_yf_1007_10000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/further_llama_7B_math_yf_1007/10000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/",
        model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/configs/further_llama_7B_math_yf.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

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


# for item in range(0, 18):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v15_{500 + item*500}",
#         type=InternLM,
#         path=f"/mnt/petrelfs/share/songdemin/model/7B_llama2_coder_sft_v15/{500 + item*500}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v15.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)