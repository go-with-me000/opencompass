# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models.internal import  InternLM

models = [
    dict(
        abbr="Newton_20B_0_4_3_3000",
        type=InternLM,
        path="/fs-computility/llm/shared/zhangshuo/ckpts/Newton_20B_0.4.3/3000",
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path="/fs-computility/llm/shared/zhangshuo/train_internlm/",
        model_config="/fs-computility/llm/chenkeyu1/program/opencompass/configs/models/model_configs/Newton_20B_0.4.1.py",
        model_type="LLAMA_TF32",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),
]
