# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models import LLama, InternLM, LLMv3

models = [
    # dict(
    #     type=InternLM,
    #     abbr="1b_baijuyi_pos_cn_en_tf32_2_60000",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/1b_baijuyi_pos_cn_en_tf32_2/60000",
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
    #     abbr="1b_baijuyi_pos_cn_en_tf32_2_70000",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/1b_baijuyi_pos_cn_en_tf32_2/70000",
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
    #     abbr="further_llama_7B_cot_830_3——100",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_cot_830_3/100/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/chenkeyu1/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/further_llama_7B_cot.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     type=InternLM,
    #     abbr="further_llama_7B_cot_830_3——200",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_cot_830_3/200/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/chenkeyu1/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/further_llama_7B_cot.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     type=InternLM,
    #     abbr="further_llama_7B_cot_830_3——300",
    #     path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_cot_830_3/300/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/chenkeyu1/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/further_llama_7B_cot.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr="mix_999",
        type=LLMv3,
        path="s3://model_weights/0331/mixv6_wm/999/",
        tokenizer_path='/mnt/petrelfs/share_data/llm_data/tokenizers/V7.model',
        model_type="origin",
        tokenizer_type='v7',
        max_out_len=100,
        max_seq_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=1)),
]
