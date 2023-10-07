from opencompass.models.internal import InternLM

models = [

    dict(
        abbr="1B_v3_28000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/wangbo/1B_v3/28000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
        model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    #
    # dict(
    #     abbr="0d02468_Newton_2795",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="/mnt/petrelfs/share_data/gaoyang/ckpts/0.4_Newton_20B_0.4.0@6500_toolbench_safety_0d02468/2795",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/share_data/yanhang/issue/20230921/tf32/",
    #     model_config="/mnt/petrelfs/share_data/yanhang/issue/20230921/a550b8122c1666e19a4a7a54991d056fe7c3e57e/configs/Newton_20B_a550b8122c1666e19a4a7a54991d056fe7c3e57e.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=4, num_procs=4)
    # ),
]
