from opencompass.models import HuggingFaceCausalLM
from opencompass.models import LLama
from opencompass.models.pjlm import LLMv3



models = [
    dict(
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








    # dict(
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/chenkeyu1/models/evaluation/falcon-7b",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           use_fast=False, ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto'),
    #     batch_padding=True,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

]
