from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="7B_llama_code_43000",
    #     type=InternLM,
    #     path="s3://checkpoints_ssd_02/songdemin/7B_llama_coder/43000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
    #     model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_code.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr="7B_pkt_79000",
        type=InternLM,
        path="s3://checkpoints_ssd_02/pengrunyu/0819/model_ckpt/further_llama2_pkt/78999/",
        tokenizer_path='/mnt/petrelfs/share_data/llm_model/tokenizer/llamav4.model',
        tokenizer_type='v4',
        module_path="/mnt/petrelfs/share_data/pengrunyu/train_internlm/",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/further_pkt.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]
