import json
import torch
from tqdm import tqdm

import io, os, sys, re

from opencompass.utils.internal.model.origin_llama.packed_pipeline_flash_origin_llama1d import Packed_Flash_Origin_LLAMA_exlarge_pipeline_1D
from opencompass.utils.internal.model.converted_llama.packed_pipeline_flash_converted_llama1d import Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D
from opencompass.utils.internal.model.scale_llama_v2.packed_pipepline_flash_scale_llama1d import Packed_Flash_scale_LLAMA_v2_exlarge_pipeline_1D
from opencompass.utils.internal.train.mask_generate import generate_scaleup_mask, get_hidden_dim

def model_name2tokenizer_info(model_name):
    if model_name in ['7132v2', 'scale100bv2']:
        return '/mnt/petrelfs/share_data/llm_weight/final_model_v6.model', 'v6'
    elif model_name in ['1005', '1006', '7132m', '7132k']:
        return '/mnt/petrelfs/share_data/llm_data/tokenizers/llamav4.model', 'v4'
    else:
        return '/mnt/petrelfs/share_data/llm_llama/tokenizer.model', 'llama'
    
maxdim2oridim = {
    5120:4096,
    10240:8192,
}

maxlay2orilay = {
    82:80,
    88:80,
    40:32,
}

def model_name2iter_info(model_name):
    if model_name in ['7132v2', 'scale100bv2', '1005', '1006', '7132m', '7132k']:
        return 500, 4000
    else:
        return 0, 0
    
model_name2model_size = {
    '7132v2':'7132',
    'scale100bv2':'1002',
    '1005':'1005+',
    '1006':'1005+',
    'finetune2':'finetune2',
    '7132m':'7132',
    '7132k':'7132',
}
    
def model_name2module(model_name):
    if 'origin' in model_name:
        return Packed_Flash_Origin_LLAMA_exlarge_pipeline_1D
    if model_name in ['7132v2', 'scale100bv2', '1005', '1006', '7132m', '7132k']:
        return Packed_Flash_scale_LLAMA_v2_exlarge_pipeline_1D
    return Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D

def convert2run(model_config):
    model_config['dtype'] = torch.half if str(model_config['dtype']) == 'torch.float16' else torch.bfloat16
    model_config['parallel_output'] = False
    return model_config
    
def convert2save(model_config):
    model_config['dtype'] = str(model_config['dtype'])
    model_config['parallel_output'] = False
    return model_config

basic_config = dict(
    num_chunks=1, checkpoint=False, dtype=torch.half, embed_split_hidden=False,
    num_layers=40, hidden_size=5120, vocab_size=150494, embed_grad_scale=1,
    parallel_output=False, num_attention_heads=40, mlp_ratio=8/3, apply_post_layer_norm=False,
    no_bias=True, deepnorm=False, residual_in_fp32=False,
    norm_type='rmsnorm', drop_rate=0, attn_drop_rate=0, model_type='llama'
)

def gen_masks(cur_iter, maxdim, oridim, maxlay, orilay, warm_iter, max_iter):
    feat_mask = generate_scaleup_mask(
        cur_iter=max_iter if cur_iter>max_iter else cur_iter,
        warumup_iter=warm_iter,
        max_iter=max_iter,
        ori_ch=oridim,
        max_ch=maxdim,
        scaleup_ch=maxdim,
        mode='v3'
    )[0].cuda()
    ffn_mask = generate_scaleup_mask(
        cur_iter=max_iter if cur_iter>max_iter else cur_iter,
        warumup_iter=warm_iter,
        max_iter=max_iter,
        ori_ch=get_hidden_dim(oridim),
        max_ch=get_hidden_dim(maxdim),
        scaleup_ch=get_hidden_dim(maxdim),
        mode='v3'
    )[0].cuda()
    layer_mask = generate_scaleup_mask(
        cur_iter=max_iter if cur_iter>max_iter else cur_iter,
        warumup_iter=warm_iter,
        max_iter=max_iter,
        ori_ch=orilay,
        max_ch=maxlay,
        scaleup_ch=maxlay,
        mode='v3'
    )[0].cuda().view(1, -1)
    return feat_mask, ffn_mask, layer_mask
    
from_locals = ['dim', 'n_heads', 'n_layers', 'vocab_size']
to_modelcnf = ['hidden_size', 'num_attention_heads','num_layers', 'vocab_size']

def local_config_convert(model_config):
    config = {}
    for oldk, newk in zip(from_locals, to_modelcnf):
        config[newk] = model_config[oldk]
    return config