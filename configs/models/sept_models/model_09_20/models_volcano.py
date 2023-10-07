# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models.internal import  InternLM

models = [
    dict(
        abbr="Qiansanqiang_7B_1.0.0_79000",
        type=InternLM,
        path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_1.0.0/79000",
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path="/fs-computility/llm/shared/zhangshuo/train_internlm/",
        model_config="/fs-computility/llm/shared/zhangshuo/train_internlm/configs/Qiansanqiang_7B_1_0_0.py",
        model_type="LLAMA_TF32",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=2, num_procs=2)
    ),

]
