# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama

from opencompass.models.internal import LLMv3, InternLM

models = [
    # dict(
    #     type=LLMv3,
    #     path="s3://checkpoints_ssd_02/pengrunyu/0819/model_ckpt/further_llama2_pkt/79000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4', model_type="converted",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)),

    dict(
        type=InternLM,
        abbr="1b_baijuyi_pos_cn_en_tf32_2_110000",
        path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/1b_baijuyi_pos_cn_en_tf32_2/110000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        module_path="/mnt/petrelfs/share_data/feizhaoye/InternLM/",
        model_type="LLAMA2",
        model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/1b_baijuyi_pos_cn_en_tf32_2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

]
