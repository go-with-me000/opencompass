from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    # dict(
    #     abbr="7B_averaged",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/0831/7B_averaged/16000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/chenkeyu1/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/InternLM/configs/7B_averaged.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr="7B_tokenizer_v10_10000",
        type=InternLM,
        path="s3://checkpoints_ssd_02/0831/7B_tokenizer_v10/10000/",
        tokenizer_path='/mnt/petrelfs/share_data/llm_model/tokenizer/v10.model',
        tokenizer_type='v7',
        module_path="/mnt/petrelfs/chenkeyu1/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_tokenizer_v10.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),

]
