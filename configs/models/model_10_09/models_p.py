from opencompass.models.internal import InternLM

models = [
    dict(
        abbr="1b_v6_28000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v6/28000",
        tokenizer_path='/mnt/petrelfs/share_data/wangbo/wiki_9998_small.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1B_v6.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="1b_v6_32000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v6/32000",
        tokenizer_path='/mnt/petrelfs/share_data/wangbo/wiki_9998_small.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1B_v6.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]