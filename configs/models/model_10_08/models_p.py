from opencompass.models.internal import InternLM

models = [
    # dict(
    #     abbr="1b_v4_28000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v4/28000",
    #     tokenizer_path='/mnt/petrelfs/share_data/wangbo/llama_small_V2.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1B_v4.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="1b_v4_32000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v4/32000",
    #     tokenizer_path='/mnt/petrelfs/share_data/wangbo/llama_small_V2.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1B_v4.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     abbr="oppenheimer_7B_0.4.1_3000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.4.1/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.4.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0.5.1_1000",
    #     type=InternLM,
    #     model_type="LLAMA_TF32",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.5.1/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.5.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0.5.1_2000",
    #     type=InternLM,
    #     model_type="LLAMA_TF32",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.5.1/2000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.5.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0.5.1_3000",
    #     type=InternLM,
    #     model_type="LLAMA_TF32",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.5.1/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.5.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    #
    # dict(
    #     abbr="oppenheimer_7B_0.6.1_1000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.6.1/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.6.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0.6.1_2000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.6.1/2000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.6.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0.6.1_3000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.6.1/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.6.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    #
    # dict(
    #     abbr="oppenheimer_7B_0.7.1_1000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.7.1/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.7.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0.7.1_1750",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.7.1/1750",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.7.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),

    dict(
        abbr="1b_v5_28000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v5/28000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
        tokenizer_type='v4',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1B_v5.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]
#
# for item in range(0, 5):
#     model_info = dict(
#         abbr=f"bohr_v0.1.0_{12000 + item * 2000}",
#         type=InternLM,
#         model_type="LLAMA",
#         path=f"/mnt/petrelfs/share_data/shaoyunfan/ckpt/bohr_0927/{12000 + item * 2000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
#         model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/bohr.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)

#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v12_{100 + item*100}",
#         type=InternLM,
#         path=f"/mnt/petrelfs/share/songdemin/model7B_llama2_coder_sft_v12/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v12.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 35):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_v3_{130000 + item*2000}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_v3/{130000 + item*2000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_v3.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
