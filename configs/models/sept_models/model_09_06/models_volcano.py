# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models.internal import  InternLM

models = [
    dict(
        type=InternLM,
        abbr="7B_maibao_safe_34000",
        path="/fs-computility/llm/shared/llm_data/ckpts/7B_maibao_safe/34000",
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path="/fs-computility/llm/shared/gaoyang/code/train_internlm/",
        model_config="/fs-computility/llm/chenkeyu1/train_internlm/configs/7B_maibao_safe.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]