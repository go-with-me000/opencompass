from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama

models = [
    # dict(
    #     type=InternLM,
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_dufu_2/18000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr="qwen-7b",
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/huggingface/Qwen-7B/",
        tokenizer_path='Qwen/Qwen-7B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        batch_padding=False,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    #
    # dict(abbr="LLama2-13B",
    #      type=LLama, path='/mnt/petrelfs/share_data/llm_llama/llama2_raw/llama-2-13b',
    #      tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
    #      max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=2, num_procs=2)),
    #

# dict(abbr="LLama13B",
#          type=LLama, path='/mnt/petrelfs/share_data/llm_llama/13B',
#          tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
#          max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=2, num_procs=2)),
    # dict(abbr="LLama30B",
    #      type=LLama, path='/mnt/petrelfs/share_data/llm_llama/30B',
    #      tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
    #      max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=4, num_procs=4)),

]
