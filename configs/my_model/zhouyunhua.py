from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv3

models = [
    dict(
        type=LLMv3,
        path="s3://model_weights/zhouyunhua/7B_kaoshi_7_5/2387",
        tokenizer_path='/mnt/petrelfs/share_data/llm_data/tokenizers/V7.model',
        model_type="origin",
        tokenizer_type='v7',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]
