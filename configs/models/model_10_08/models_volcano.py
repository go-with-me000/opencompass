# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models.internal import  InternLM

models = [
    dict(
        abbr='Qiansanqiang_7B_5.0.3_84000',
        type='opencompass.models.internal.InternLM',
        path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_5.0.3/84000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/zhangshuo/develop/',
        model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_7B_5_0_3.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    # dict(
    #     abbr='Qiansanqiang_7B_5.0.4_84000',
    #     type='opencompass.models.internal.InternLM',
    #     path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_5.0.4/84000',
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
    #     tokenizer_type='llama',
    #     module_path='/fs-computility/llm/shared/zhangshuo/develop/',
    #     model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_7B_5_0_4.py',
    #     model_type='LLAMA',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr='Qiansanqiang_7B_5.0.5_84000',
    #     type='opencompass.models.internal.InternLM',
    #     path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_5.0.5/84000',
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
    #     tokenizer_type='llama',
    #     module_path='/fs-computility/llm/shared/zhangshuo/develop/',
    #     model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_7B_5_0_5.py',
    #     model_type='LLAMA',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

]
