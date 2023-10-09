from opencompass.models import HuggingFaceCausalLM
from opencompass.models.internal import  InternLM

models = [

]

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

for item in range(0, 16):
    model_info = dict(
        abbr=f"7B_llama2_coder_sft_v8_{100 + item*100}",
        type=InternLM,
        path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v8/{100 + item*100}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v7.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)
