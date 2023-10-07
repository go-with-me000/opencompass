from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3

models = [
    dict(
        type=InternLM,
        path="/cpfs01/shared/alillm2/alillm2_hdd/zhangshuo/ckpt/0831/20B_newton_stage2/69000/",
        tokenizer_path='/cpfs01/shared/public/tokenizers/V7.model',
        tokenizer_type='v7',
        model_config="/cpfs01/user/chenkeyu1/program/opencompass/InternLM/configs/20B_newton_stage2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),
]
