from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    #  dict(
    #     abbr="7b_baijuyi_v3_60000",
    #     type=InternLM,
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/{JOB_NAME}/60000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="7b_baijuyi_v3_72000",
    #     type=InternLM,
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/{JOB_NAME}/72000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        type=InternLM,
        path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_dufu_2/18000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    #
    dict(
        type=InternLM,
        path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_baijuyi_2/24000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),


    dict(
        type=InternLM,
        path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_refineweb_2/19000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

    # dict(
    #     type=InternLM,
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/1b_baijuyi_pos_fp32emb/17000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/InternLM/configs/1b_baijuyi_pos_fp32emb.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     type=LLMv3,
    #     path="s3://model_weights/0331/finetune7b/12000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     model_type="converted",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)),

]
