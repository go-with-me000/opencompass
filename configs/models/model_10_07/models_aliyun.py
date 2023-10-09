from opencompass.models.internal import InternLM, LLMv3, LLama

models = [
    dict(
        abbr="qianxuesen_3B_v0.1.0_140000",
        type=InternLM,
        path="/cpfs01/shared/public/lvhaijun/ckpts/qianxuesen_3B_v0.1.0/140000",
        tokenizer_path='/cpfs01/shared/public/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path="/cpfs01/shared/public/lvhaijun/train_internlm_stream/",
        model_config="/cpfs01/shared/public/lvhaijun/train_internlm_stream/configs/qianxuesen_3B_v0.1.0.py",
        model_type="LLAMA",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
]
