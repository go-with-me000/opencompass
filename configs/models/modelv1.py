from opencompass.models import HuggingFaceCausalLM
from opencompass.models import LLama

models = [
    dict(abbr="LLama7B",
         type=LLama, path='/mnt/petrelfs/share_data/llm_llama/7B',
         tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
         max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1, num_procs=1)),
    dict(abbr="LLama13B",
         type=LLama, path='/mnt/petrelfs/share_data/llm_llama/13B',
         tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
         max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=2, num_procs=2)),

    dict(abbr="LLama2-7B",
         type=LLama, path='/mnt/petrelfs/share_data/llm_llama/llama2_raw/llama-2-7b',
         tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
         max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1, num_procs=1)),
    dict(abbr="LLama2-13B",
         type=LLama, path='/mnt/petrelfs/share_data/llm_llama/llama2_raw/llama-2-13b',
         tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
         max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=2, num_procs=2)),
    #
    dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/evaluation/open_llama_3b",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/evaluation/open_llama_7b",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto', ),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/evaluation/open_llama_13b",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=2, num_procs=2),
    ),
    dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/evaluation/open_llama_3b_v2",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/evaluation/open_llama_7b_v2",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        path="/mnt/petrelfs/chenkeyu1/models/evaluation/mpt-7b",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False, ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto',trust_remote_code=True),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),








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
