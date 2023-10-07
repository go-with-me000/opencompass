from opencompass.models.internal import  InternLM

models = [
    dict(
        abbr="copernicus_7b_v0_0_1_100",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/copernicus_7b_v0.0.1/100",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/feizhaoye/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/copernicus_7b_v0_0_1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="copernicus_7b_v0_0_1_200",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/copernicus_7b_v0.0.1/200",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/feizhaoye/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/copernicus_7b_v0_0_1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="copernicus_7b_v0_0_1_300",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/copernicus_7b_v0.0.1/300",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/feizhaoye/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/copernicus_7b_v0_0_1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

#
# dict(
#         abbr="copernicus_7b_v0_0_2_100",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0_0_2/100",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share_data/feizhaoye/train_internlm/",
#         model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/copernicus_7b_v0_0_2.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
#     dict(
#         abbr="copernicus_7b_v0_0_2_200",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0_0_2/200",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share_data/feizhaoye/train_internlm/",
#         model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/copernicus_7b_v0_0_2.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
#     dict(
#         abbr="copernicus_7b_v0_0_2_300",
#         type=InternLM,
#         model_type="LLAMA",
#         path="s3://checkpoints_ssd_02/copernicus_7b_v0_0_2/300",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share_data/feizhaoye/train_internlm/",
#         model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/copernicus_7b_v0_0_2.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),

    # dict(
    #     abbr="open_llama_code_3B_105000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/3B_open_llama_coder/105000/",
    #     tokenizer_path='/mnt/petrelfs/share/songdemin/code/download_model/open_llama_3b_v2/tokenizer.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/open_llama_code_3B.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
]
