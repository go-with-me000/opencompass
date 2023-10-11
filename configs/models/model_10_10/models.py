# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama

from opencompass.models.internal import InternLM, LLMv3

models = [

    dict(
        abbr="mabi_bao_999",
        type=LLMv3,
        path="/mnt/petrelfs/share_data/wangguoteng.p/maibao_999",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        model_type="origin",
        tokenizer_type='v7',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]
