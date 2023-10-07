from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    dict(
        type=InternLM,
        path="s3://checkpoints_ssd_02/0831/7B_cluster_debug/snapshot/0/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/InternLM/configs/7B_cluster_debug.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),



]
