
from opencompass.models.collie_model_wrapper import CollieModel
from opencompass.models.hf_model_wrapper import HuggingFaceFEPECausalLM

import numpy as np

group = 'pjlab_fepe_llama2_7B_4096'

tags = [('hang_500', {'exp': False, '1d': False, 'imp': False, 'log': False, 
                      'base': 500.0, 'exp_base': 512.0, 'log_base': 2048, 
                      'ntk_option': 'none', 'ntk_alpha': 1.0, 'pi_lambda': 1.0,
                      'max_length': 4096, 'interleave': True, 'hf_interleave': True}), 
        ('hang_1000', {'exp': False, '1d': False, 'imp': False, 'log': False, 
                       'base': 1000.0, 'exp_base': 512.0, 'log_base': 2048, 
                       'ntk_option': 'none', 'ntk_alpha': 1.0, 'pi_lambda': 1.0,
                       'max_length': 4096, 'interleave': True, 'hf_interleave': True}),
        ('rope_inv_2d_raw', {'exp': False, '1d': False, 'imp': False, 'log': False, 
                             'base': 10000.0, 'exp_base': 512.0, 'log_base': 2048, 
                             'ntk_option': 'none', 'ntk_alpha': 1.0, 'pi_lambda': 1.0,
                             'max_length': 4096, 'interleave': True, 'hf_interleave': True}), 
        ('rope_inv_2d_raw_pi_2', {'exp': False, '1d': False, 'imp': False, 'log': False, 
                                  'base': 10000.0, 'exp_base': 512.0, 'log_base': 2048, 
                                  'ntk_option': 'none', 'ntk_alpha': 1.0, 'pi_lambda': 2.0,
                                  'max_length': 4096, 'interleave': True, 'hf_interleave': True}), ]
             
models = [dict(
            abbr='{}-{}'.format(group, tag),  # name in path
            type=HuggingFaceFEPECausalLM,
            model_path='p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-{}/epoch_1/pytorch_model.bin'.format(group, tag),
            path='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                                  trust_remote_code=True, use_fast=False, ),
            pe_config=pe_config,
            max_out_len=128,
            max_seq_len=2048,
            batch_size=8,
            batch_padding=True,  # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=1, num_procs=1),
        ) for tag, pe_config in tags]

"""
tags = ['rope_inv_2d_raw', 'rope_inv_2d_log', 'rope_inv_1d_raw', 'rope_inv_1d_log', 
        'rope_imp_2d_raw', 'rope_imp_2d_log', 'rope_imp_1d_raw', 'rope_imp_1d_log', 
        'xpos_inv_2d_raw', 'xpos_inv_2d_log', 'xpos_inv_1d_raw', 'xpos_inv_1d_log', 
        'xpos_imp_2d_raw', 'xpos_imp_2d_log', 'xpos_imp_1d_raw', 'xpos_imp_1d_log', ]
        
def tag_to_pe_config(tag):
    return {'exp': tag.__contains__('xpos'), '1d': tag.__contains__('1d'), 
            'imp': tag.__contains__('imp'), 'log': tag.__contains__('log'), 
            'exp_base': 512.0, 'log_base': 2048.0, 'ntk_option': 'none', 'ntk_alpha': 1.0,
            'max_length': 8192, 'interleave': True, 'hf_interleave': True}
             
models = [dict(
            abbr='{}-{}'.format(group, tag),  # name in path
            type=HuggingFaceFEPECausalLM,
            model_path='p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-{}/epoch_1/pytorch_model.bin'.format(group, tag),
            path='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                                  trust_remote_code=True, use_fast=False, ),
            pe_config=tag_to_pe_config(tag),
            max_out_len=128,
            max_seq_len=2048,
            batch_size=8,
            batch_padding=True,  # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=1, num_procs=1),
        ) for tag in tags]
"""

"""
    pe_config: dict,
    model_path: str, 
    config_path: str,
    tokenizer_kwargs: dict = dict(),
    meta_template: Optional[Dict] = None,
    extract_pred_after_decode: bool = False,
    batch_padding: bool = False
"""

# models = [
#     # dict(
#     #     abbr='{}-{}'.format(group, tag),  # name in path
#     #     type=CollieModel,
#     #     pe_config=pe_config,
#     #     model_path='p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-{}/epoch_1/'.format(group, tag),
#     #     config_path='/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf/',
#     #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
#     #                           trust_remote_code=True, use_fast=False, ),
#     #     max_out_len=8,
#     #     max_seq_len=2048,
#     #     batch_size=16,
#     #     batch_padding=True,  # if false, inference with for-loop without batch padding
#     #     run_cfg=dict(num_gpus=1, num_procs=1),
#     # ),
#     dict(
#         abbr='{}-{}'.format(group, tag),  # name in path
#         type=HuggingFaceFEPECausalLM,
#         model_path='p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-{}/epoch_1/pytorch_model.bin'.format(group, tag),
#         path='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',
#         tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
#                               trust_remote_code=True, use_fast=False, ),
#         pe_config=pe_config,
#         max_out_len=128,
#         max_seq_len=2048,
#         batch_size=8,
#         batch_padding=True,  # if false, inference with for-loop without batch padding
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     ),
# ]
