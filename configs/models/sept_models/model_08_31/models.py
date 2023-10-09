# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models import LLama
models = [
    # dict(
    #     type=InternLM,
    #     abbr="further_llama_7B_cot_3000",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_cot_2/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/chenkeyu1/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/further_llama_7B_cot.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(abbr="LLama7B",
         type=LLama, path='/mnt/petrelfs/share_data/llm_llama/7B',
         tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
         max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1, num_procs=1)),
]
