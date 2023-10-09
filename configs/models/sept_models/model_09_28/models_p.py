from opencompass.models import HuggingFaceCausalLM
from opencompass.models.internal import  InternLM

models = [
# dict(
#         abbr="oppenheimer_1B_0_0_1_20000",
#         type=InternLM,
#         model_type="LLAMA_TF32",
#         path="s3://checkpoints_ssd_02/oppenheimer/oppenheimer_1B_0.0.1/20000",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
#         tokenizer_type='v7',
#         module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/",
#         model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.1.2/configs/oppenheimer_1B_0.0.1.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     ),
]

# for item in range(0, 1):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v5_{100 + item*100}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v5/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v5.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v6_{100 + item*100}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v6/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v6.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v7_{100 + item*100}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v7/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v7.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v8_{100 + item*100}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v8/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v8.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v9_{100 + item*100}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v9/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v9.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)
#
# for item in range(0, 20):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v10_{100 + item*100}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v10/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v10.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)

for item in range(0, 20):
    model_info = dict(
        abbr=f"7B_llama2_coder_sft_v11_{100 + item*100}",
        type=InternLM,
        path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v11/{100 + item*100}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v11.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)
