from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    dict(
        type=InternLM,
        path="s3://model_weights/0831/feizhaoye/3B_baijuyi_v2/25000/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    # dict(
    #
    #     type=InternLM,
    #     path="s3://model_weights/0831/feizhaoye/7B_baijuyi_v3/10000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     type=InternLM,
    #     path="s3://model_weights/0831/feizhaoye/1b_baijuyi_posneg/16000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     type=InternLM,
    #     path="s3://model_weights/0831/feizhaoye/1b_baijuyi_pos/18000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     type=LLMv3,
    #     path="s3://model_weights/0331/linglongta_v4_2_cn_v3/8999/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4', model_type="origin",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)),
    # dict(
    #     type=LLMv3,
    #     path="s3://model_weights/0331/linglongta_v4_2_cn_v3/7999/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4', model_type="origin",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)),
    # dict(
    #     type=LLMv3,
    #     path="s3://model_weights/0331/linglongta_v4_2_cn_v3/6999/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4', model_type="origin",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)),
]