from opencompass.models import HuggingFaceCausalLM
from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="oppenheimer_7B_0_1_1_3000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.1.1/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.1.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0_2_1_3000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.2.1/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    #     tokenizer_type='v4',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.2.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0_3_1_3000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.3.1/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.3.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="oppenheimer_7B_0_4_1_2000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/oppenheimer_7B_0.4.1/2000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_7B_0.4.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="GAIRMath_Abel-7b",
    #     type=HuggingFaceCausalLM,
    #     path="GAIR/GAIRMath-Abel-7b",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False, max_length=2048),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto',trust_remote_code=True),
    #     # batch_padding=True,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
    #
    # dict(
    #     abbr="MAmmoTH-7B_0",
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/share_data/feizhaoye/huggingface/mammoth",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False, max_length=2048),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     # batch_padding=True,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),

    dict(
        abbr=f"further_llama_7B_math_yf_3000",
        type=InternLM,
        path=f"s3://checkpoints_ssd_02/further_llama_7B_math_yf_0925/3000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
        model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_yf_eval.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    # dict(
    #     abbr="baichuan2_further_train_v1_19500",
    #     type=InternLM,
    #     model_type="BAICHUAN2",
    #     path="/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v1_19500/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
    #     tokenizer_type='baichuan2',
    #     module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="baichuan2_further_trainv0_22000",
    #     type=InternLM,
    #     model_type="BAICHUAN2",
    #     path="/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v0_22000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
    #     tokenizer_type='baichuan2',
    #     module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

]
#
# for item in range(0, 5):
#     model_info = dict(
#         abbr=f"metamath_llama2_7b_{200 + item*200}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/further_llama_7B_math_cot_0918/{200 + item*200}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
#         model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_cot_eval.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)

# for item in range(0, 16):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_{100000 + item*2000}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_v3/{100000 + item*2000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_v3.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     if item in [0,14,15]:
#         models.append(model_info)
