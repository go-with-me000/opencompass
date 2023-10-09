from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="3B_open_llama_code_token_10B_129000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/3B_open_llama_coder/129000/",
    #     tokenizer_path='/mnt/petrelfs/share/songdemin/code/download_model/open_llama_3b_v2/tokenizer.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/open_llama_code_3B.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="7B_llama_coder_python_16B_token_2000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/2000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python_16B_token.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
]

for item in range(1, 21):
    model_info = dict(
        abbr=f"copernicus_7b_v0_0_1_{item*100}",
        type=InternLM,
        model_type="LLAMA",
        path=f"s3://checkpoints_ssd_02/copernicus_7b_v0.0.1/{item*100}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/feizhaoye/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/copernicus_7b_v0_0_1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)
