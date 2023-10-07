from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="LLAMA-MTP-7B-EN_3000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/pengrunyu/demo_7B_mtp_en/3000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/pengrunyu/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/models/model_configs/LLAMA-MTP-7B.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr="7B_llama_coder_python_22000",
        type=InternLM,
        path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/22000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="7B_llama_coder_python_24000",
        type=InternLM,
        path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/24000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),


]

for item in range(0, 7):
    model_info = dict(
        abbr=f"7B_llama_coder_{36000+item*2000}",
        type=InternLM,
        path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_v3/{36000+item*2000}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)
