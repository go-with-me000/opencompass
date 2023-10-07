from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama

models = [
    dict(
        type=InternLM,
        abbr="further_llama_7B_cot_v5_3_400",
        path="boto3:s3://hdd_new_model_weights.10.140.31.252/0831/feizhaoye/further_llama_7B_cot_v5_3/400",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/feizhaoye/InternLM/",
        model_type="LLAMA2",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/further_llama_7B_cot.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

    # dict(
    #     type=InternLM,
    #     abbr="1b_baijuyi_pos_cn_en_tf32_2_50000",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/1b_baijuyi_pos_cn_en_tf32_2/50000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/share_data/feizhaoye/InternLM/",
    #     model_type="LLAMA2",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/1b_baijuyi_pos_cn_en_tf32_2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     type=InternLM,
    #     abbr="further_llama_7B_math_yf_1000",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_math_yf_2/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/share_data/feizhaoye/InternLM/",
    #     model_type="LLAMA2",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/further_llama_7B_math_yf.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

]
