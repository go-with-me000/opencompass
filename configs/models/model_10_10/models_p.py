from opencompass.models import HuggingFaceCausalLM
from opencompass.models.internal import InternLM

models = [
#     dict(
#         abbr="baichuan_train_7b_base_v2_50000",
#         type=InternLM,
#         model_type="BAICHUAN2",
#         path="/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v2/50000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
#         tokenizer_type='baichuan2',
#         module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm",
#         model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
#     dict(
#         abbr="baichuan_train_7b_base_v2_52000",
#         type=InternLM,
#         model_type="BAICHUAN2",
#         path="/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v2/52000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
#         tokenizer_type='baichuan2',
#         module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm",
#         model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
# dict(
#         abbr="baichuan_train_7b_base_v4_50000",
#         type=InternLM,
#         model_type="BAICHUAN2",
#         path="/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v4/50000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
#         tokenizer_type='baichuan2',
#         module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm",
#         model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
#     dict(
#         abbr="baichuan_train_7b_base_v4_52000",
#         type=InternLM,
#         model_type="BAICHUAN2",
#         path="/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v4/52000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
#         tokenizer_type='baichuan2',
#         module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm",
#         model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
# dict(
#         abbr="Mistral-7B",
#         type=HuggingFaceCausalLM,
#         path="/mnt/petrelfs/share_data/chenkeyu1/models/huggingface/Mistral-7B-v0.1",
#         tokenizer_kwargs=dict(padding_side='left',
#                               truncation_side='left',
#                               trust_remote_code=True,
#                               # use_fast=False,
#                               ),
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         model_kwargs=dict(device_map='auto',
#                           trust_remote_code=True,
#                           ),
#         batch_padding=True,
#         pad_token_id=0,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     ),
]
# for item in range(0, 7):
#     model_info = dict(
#         abbr=f"bohr_v0.2.0_{2000 + item * 2000}",
#         type=InternLM,
#         model_type="LLAMA",
#         path=f"/mnt/petrelfs/share_data/shaoyunfan/ckpt/bohr_1008/{2000 + item * 2000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
#         model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm_1007/configs/bohr.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v12_random_{9000 + item*1000}",
#         type=InternLM,
#         path=f"/mnt/petrelfs/share/songdemin/model/7B_llama2_coder_sft_v12_random/{9000 + item*1000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v12_random.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
#
# for item in range(0, 10):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v15_{9000 + item*1000}",
#         type=InternLM,
#         path=f"/mnt/petrelfs/share/songdemin/model/7B_llama2_coder_sft_v15/{9000 + item*1000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="//mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v15.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v13_{1000 + item*1000}",
#         type=InternLM,
#         path=f"/mnt/petrelfs/share/songdemin/model/7B_llama2_coder_sft_v13/{1000 + item*1000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v13.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)

for item in range(0, 48):
    model_info = dict(
        abbr=f"7B_llama2_coder_sft_v16_{100 + item*100}",
        type=InternLM,
        path=f"/mnt/petrelfs/share/songdemin/model/7B_llama2_coder_sft_v16/{100 + item*100}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v16.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)