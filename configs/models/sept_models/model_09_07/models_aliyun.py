from opencompass.models.internal import InternLM

models = [
    dict(
        abbr = "20B_newton_cot_qa_superglue_sft_130000",
        type=InternLM,
        path="/cpfs01/shared/alillm2/alillm2_hdd/zhangshuo/ckpt/0831/20B_newton_stage2_0905/130000",
        tokenizer_path='/cpfs01/shared/public/tokenizers/V7.model',
        tokenizer_type='v7',
        module_path="/cpfs01/shared/alillm2/user/zhangshuo/InternLM/",
        model_config="/cpfs01/user/chenkeyu1/train_internlm/configs/20B_newton_stage2.py",
        model_type="LLAMA2",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),
]