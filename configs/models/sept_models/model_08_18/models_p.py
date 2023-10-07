from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    dict(
        type=InternLM,
        path="s3://checkpoints_ssd_02/0831/20B_newton_stage2/36000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        model_config="/mnt/petrelfs/chenkeyu1/program/opencompass/InternLM/configs/20B_newton_stage2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),

]
