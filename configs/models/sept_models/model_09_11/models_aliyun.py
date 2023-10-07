from opencompass.models.internal import InternLM

models = [
    dict(
        abbr = "Newton_20B_0.4.1_28500",
        type=InternLM,
        path="/cpfs01/shared/public/zhangshuo/ckpts/Newton_20B_0.4.1/28500",
        tokenizer_path='/cpfs01/shared/public/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path="/cpfs01/shared/public/zhangshuo/train_internlm/",
        model_config="/cpfs01/user/chenkeyu1/train_internlm/configs/Newton_20B_0.5.0.py",
        model_type="LLAMA_TF32",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),
]
