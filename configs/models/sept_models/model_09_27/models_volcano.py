# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="Qiansanqiang_7B_2.0.0_22000",
    #     type=InternLM,
    #     path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_2.0.0/22000",
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/fs-computility/llm/shared/zhangshuo/train_internlm/",
    #     model_config="/fs-computility/llm/shared/zhangshuo/train_internlm/configs/Qiansanqiang_7B_1_0_0.py",
    #     model_type="LLAMA_TF32",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="Qiansanqiang_7B_3.0.0_22000",
    #     type=InternLM,
    #     path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_3.0.0/22000",
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/fs-computility/llm/shared/zhangshuo/train_internlm/",
    #     model_config="/fs-computility/llm/shared/zhangshuo/train_internlm/configs/Qiansanqiang_7B_1_0_0.py",
    #     model_type="LLAMA_TF32",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=2, num_procs=2)
    # ),
    # dict(
    #     abbr="Qiansanqiang_1B_4.0.1_22000",
    #     type=InternLM,
    #     path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.1/22000",
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/fs-computility/llm/shared/zhangshuo/develop/",
    #     model_config="/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_1.py",
    #     model_type="INTERNLM",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    #
    # dict(
    #     abbr="Qiansanqiang_1B_4.0.2_22000",
    #     type=InternLM,
    #     path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.2/22000",
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/fs-computility/llm/shared/zhangshuo/develop/",
    #     model_config="/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_1.py",
    #     model_type="INTERNLM",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),



dict(
        abbr="Qiansanqiang_7B_5.0.0_2000",
        type=InternLM,
        path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_5.0.0/2000",
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path="/fs-computility/llm/shared/zhangshuo/develop/",
        model_config="/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_7B_5_0_0.py",
        model_type="LLAMA",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        run_cfg=dict(num_gpus=2, num_procs=2)
    ),
dict(
        abbr="Qiansanqiang_7B_5.0.1_2000",
        type=InternLM,
        path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_7B_5.0.1/2000",
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path="/fs-computility/llm/shared/zhangshuo/develop/",
        model_config="/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_7B_5_0_0.py",
        model_type="LLAMA",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        run_cfg=dict(num_gpus=2, num_procs=2)
    ),

    # dict(
    #     abbr="Qiansanqiang_1B_4.0.1_36000",
    #     type=InternLM,
    #     path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.1/36000",
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/fs-computility/llm/shared/zhangshuo/develop/",
    #     model_config="/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_1.py",
    #     model_type="INTERNLM",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    #
    # dict(
    #     abbr="Qiansanqiang_1B_4.0.2_36000",
    #     type=InternLM,
    #     path="/fs-computility/llm/shared/zhangshuo/ckpts/Qiansanqiang_1B_4.0.2/36000",
    #     tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/fs-computility/llm/shared/zhangshuo/develop/",
    #     model_config="/fs-computility/llm/shared/zhangshuo/develop/configs/Qiansanqiang_1B_4_0_1.py",
    #     model_type="INTERNLM",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
]
