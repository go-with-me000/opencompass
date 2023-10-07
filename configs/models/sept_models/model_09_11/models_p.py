from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="copernicus_7b_v0_3_2_100",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/copernicus_7b_v0.3.2/100",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/models/model_configs/copernicus_7b_v0_0_2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="copernicus_7b_v0_3_2_500",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/copernicus_7b_v0.3.2/500",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/models/model_configs/copernicus_7b_v0_0_2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="copernicus_7b_v0_3_2_1000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/copernicus_7b_v0.3.2/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/models/model_configs/copernicus_7b_v0_0_2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     abbr="7B_llama_coder_python_12000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/12000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="7B_llama_coder_python_14000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/14000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="7B_llama_coder_python_16000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/16000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="7B_llama_coder_python_18000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/18000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="7B_llama_coder_python_20000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder_python/20000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder_python.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr="copernicus_7b_v0_1_1_100",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/copernicus_7b_v0.1.1/100",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
        model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/configs/models/model_configs/copernicus_7b_v0_0_1.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),


]
# for item in range(0, 11):
#     model_info = dict(
#         abbr=f"7B_llama_coder_{14000+item*2000}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_v3/{14000+item*2000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
