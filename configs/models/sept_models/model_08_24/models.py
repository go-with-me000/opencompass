from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    dict(
        type=InternLM,
        path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_baijuyi_2/18000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
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


    dict(
        type=InternLM,
        path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_refineweb_2/18000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

]
