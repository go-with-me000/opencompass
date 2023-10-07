from opencompass.models.internal import  InternLM

models = [
    dict(
        abbr="test_1000",
        type=InternLM,
        model_type="LLAMA",
        path="/mnt/petrelfs/llmit/ckpt/123B/sft_plato_texun_v08108rc0/11",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
        tokenizer_type='v4',
        module_path="/mnt/petrelfs/llmit/code/train_123b_8k/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/files/test/train_internlm/configs/plato_123B_8k_sft_further_train.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=8, num_procs=8)
    ),

]

#
# for item in range(0, 3):
#     model_info = dict(
#         abbr=f"oppenheimer_7B_0_0_1_{250 + item*250}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/oppenheimer_7B_0.0.1/{250 + item*250}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.0.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
# for item in range(0, 3):
#     model_info = dict(
#         abbr=f"oppenheimer_7B_0_1_1_{250 + item*250}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/oppenheimer_7B_0.1.1/{250 + item*250}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.1.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
