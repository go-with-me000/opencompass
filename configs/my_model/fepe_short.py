
from opencompass.models.collie_model_wrapper import CollieModel
from opencompass.models.hf_model_wrapper import HuggingFaceFEPECausalLM
from opencompass.models.huggingface import HuggingFaceCausalLM

import numpy as np

group = 'llama2_7B'

paths = {'llama2-7B': '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/', 
         'llama2-13B': '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-13b-hf/', }

tags = [
        # ('', 'llama2-7B', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-ntk_fixed_8', 'llama2-7B', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'fixed', 'ntk_alpha': 8., }), 
        # ('-ntk_dynamic', 'llama2-7B', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }), 
        # ('-ft', 'rope_inv_2d_raw', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-pi_2', 'rope_inv_2d_raw_pi_2', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 2, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_500_xpos_log', 'hang_500', 
        #  {'exp': True, '1d': False, 'imp': False, 'log': True, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_500', 'hang_500', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_1000_xpos_log', 'hang_1000', 
        #  {'exp': True, '1d': False, 'imp': False, 'log': True, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 1000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_1000', 'hang_1000', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 1000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_652', 'hang_652', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 652.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_2608', 'hang_2608', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 2608.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_1304', 'hang_1304', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 1304.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('', 'llama2-13B', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-ntk_fixed_8', 'llama2-13B', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'fixed', 'ntk_alpha': 8., }), 
        # ('-base_500', 'hang_500', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_500_xpos_log', 'hang_500', 
        #  {'exp': True, '1d': False, 'imp': False, 'log': True, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_652', 'hang_652', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 652.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_1000', 'hang_1000', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 1000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_1304', 'hang_1304', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 1304.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-base_2608', 'hang_2608', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 2608.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-pi_2', 'rope_inv_2d_raw_pi_2', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 2, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        # ('-ft', 'rope_inv_2d_raw', 
        #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
        #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
        #   'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
    ]

models = [
    dict(
        abbr='{}{}'.format(group, abbr),  # name in path
        type=CollieModel, 
        pe_config=pe_config,
        model_path=paths[tag] if tag in paths else 'p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/pjlab_fepe_{}_4096-{}/epoch_1/'.format(group, tag),  # pytorch_model.bin
        config_path=paths[group.replace('_', '-')],
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                              trust_remote_code=True, use_fast=False, ),
        max_out_len=128,  # 128
        max_seq_len=2048,  # 20480
        batch_size=8,  # 1
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),  # tp or pp size
    ) for abbr, tag, pe_config in tags]

# models = [
#     dict(
#         abbr='{}{}'.format(group, abbr),  # name in path
#         type=HuggingFaceCausalLM, 
#         path='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',
#         tokenizer_path='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',
#         tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
#                               trust_remote_code=True, use_fast=False, ),
#         max_out_len=128,  # 128
#         max_seq_len=2048,  # 20480
#         batch_size=8,  # 1
#         batch_padding=True,  # if false, inference with for-loop without batch padding
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     ) for abbr, tag, pe_config in tags]

# models = [dict(
#             abbr='{}-{}'.format(group, abbr),  # name in path
#             type=HuggingFaceFEPECausalLM,
#             model_path=paths[tag] if tag in paths else 'p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-{}/epoch_1/pytorch_model.bin'.format(group, tag),
#             path='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',
#             tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
#                                   trust_remote_code=True, use_fast=False, ),
#             pe_config=pe_config,
#             max_out_len=128,
#             max_seq_len=20480,  # 2048
#             batch_size=1,  # 8
#             batch_padding=True,  # if false, inference with for-loop without batch padding
#             run_cfg=dict(num_gpus=1, num_procs=1),
#         ) for abbr, tag, pe_config in tags]
