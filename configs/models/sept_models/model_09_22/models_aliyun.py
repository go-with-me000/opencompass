from opencompass.models.internal import InternLM, LLMv3

models = [
    dict(
        abbr="linglongtav14_249999",
        type=LLMv3,
        # path="/cpfs01/shared/public/users/chenkeyu1/models/linglongta_v4_2_refined_further/18999/",
        path="/cpfs01/shared/public/chenkeyu1/models/linglongta_v14/249999/",
        tokenizer_path='/cpfs01/shared/public/tokenizers/llama.model',
        tokenizer_type='v7',
        model_type="origin",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)),
]
