# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models.internal import  InternLM

models = [
    dict(
        abbr='Qiansanqiang_1B_4_0_1_48000',
        type='opencompass.models.internal.InternLM',
        path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.1/48000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path='/fs-computility/llm/shared/zhangshuo/develop/',
        model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_1.py',
        model_type='INTERNLM',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr='Qiansanqiang_1B_4_0_2_48000',
        type='opencompass.models.internal.InternLM',
        path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.2/48000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path='/fs-computility/llm/shared/zhangshuo/develop/',
        model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_2.py',
        model_type='INTERNLM',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr='Qiansanqiang_1B_4_0_3_48000',
        type='opencompass.models.internal.InternLM',
        path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.3/48000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path='/fs-computility/llm/shared/zhangshuo/develop/',
        model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_3.py',
        model_type='INTERNLM',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr='Qiansanqiang_1B_4_0_4_48000',
        type='opencompass.models.internal.InternLM',
        path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.4/48000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path='/fs-computility/llm/shared/zhangshuo/develop/',
        model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_4.py',
        model_type='INTERNLM',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr='Qiansanqiang_1B_4_0_5_48000',
        type='opencompass.models.internal.InternLM',
        path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.5/48000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path='/fs-computility/llm/shared/zhangshuo/develop/',
        model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_5.py',
        model_type='INTERNLM',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr='Qiansanqiang_7B_5_0_4_5000',
        type='opencompass.models.internal.InternLM',
        path='/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_5.0.4/5000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/zhangshuo/develop/',
        model_config='/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_7B_5_0_4.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),


]