
from opencompass.models.collie_model_wrapper import CollieModel
from opencompass.models.hf_model_wrapper import HuggingFaceFEPECausalLM

pe_config = {'exp': True, '1d': True, 'imp': True, 'log': False, 'max_length': 8192, 
             'exp_base': 512.0, 'log_base': 2048.0, 'ntk_option': 'none', 'ntk_alpha': 1.0,
             'interleave': True, 'hf_interleave': True}
group = 'pjlab_fepe_llama2_7B_4096'
tag = '{}_{}_{}_{}'.format('xpos' if pe_config['exp'] else 'rope', 
                           'imp' if pe_config['imp'] else 'inv',
                           '1d' if pe_config['1d'] else '2d', 
                           'log' if pe_config['log'] else 'raw')

"""
    pe_config: dict,
    model_path: str, 
    config_path: str,
    tokenizer_kwargs: dict = dict(),
    meta_template: Optional[Dict] = None,
    extract_pred_after_decode: bool = False,
    batch_padding: bool = False
"""

models = [
    # dict(
    #     abbr='{}-{}'.format(group, tag),  # name in path
    #     type=CollieModel,
    #     pe_config=pe_config,
    #     model_path='p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-{}/epoch_1/'.format(group, tag),
    #     config_path='/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf/',
    #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
    #                           trust_remote_code=True, use_fast=False, ),
    #     max_out_len=8,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     batch_padding=True,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
    dict(
        abbr='{}-{}'.format(group, tag),  # name in path
        type=HuggingFaceFEPECausalLM,
        # pe_config=pe_config,
        model_path='p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-{}/epoch_1/pytorch_model.bin'.format(group, tag),
        path='/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf/',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                              trust_remote_code=True, use_fast=False, ),
        pe_config=pe_config,
        max_out_len=128,
        max_seq_len=2048,
        batch_size=8,
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
