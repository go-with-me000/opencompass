from opencompass.models.internal import InternLM

models = [
    # dict(
    #     abbr="1b_v1_32000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v1/32000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="1b_v2_32000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v2/32000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="1B_v3_28000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v3/28000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/wangbo/wiki_9998.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="1B_v3_32000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v3/32000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/wangbo/wiki_9998.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    #


    dict(
        abbr="copernicus_7b_v0.5.1_1000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/copernicus_7b_v0.5.1/1000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
        model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.5.1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="copernicus_7b_v0.5.1_2000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/copernicus_7b_v0.5.1/2000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
        model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.5.1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="copernicus_7b_v0.5.1_3000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/copernicus_7b_v0.5.1/3000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
        model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.5.1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
#
#
# dict(
#         abbr="copernicus_7b_v0.6.1_1000",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0.6.1/1000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
#         tokenizer_type='v4',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.6.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
#     dict(
#         abbr="copernicus_7b_v0.6.1_2000",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0.6.1/2000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.6.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
#     dict(
#         abbr="copernicus_7b_v0.6.1_3000",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0.6.1/3000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.6.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
#
# dict(
#         abbr="copernicus_7b_v0.7.1_1000",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0.7.1/1000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.7.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
# dict(
#         abbr="copernicus_7b_v0.7.1_1750",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0.7.1/1750",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.7.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
]
#
# for item in range(0, 5):
#     model_info = dict(
#         abbr=f"metamat_full_llama2_{200 + item * 200}",
#         type=InternLM,
#         model_type="LLAMA",
#         path=f"/mnt/petrelfs/share_data/shaoyunfan/ckpt/metamath_1007/{200 + item * 200}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
#         model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_cot.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)