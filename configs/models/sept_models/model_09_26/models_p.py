from opencompass.models import HuggingFaceCausalLM
from opencompass.models.internal import  InternLM

models = [
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

    # dict(
    #     abbr=f"further_llama_7B_math_yf_5000",
    #     type=InternLM,
    #     path=f"s3://checkpoints_ssd_02/further_llama_7B_math_yf_0925/5000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_yf_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    dict(
        abbr="f61851e206f4191ba8a72b3f3dac45f17de4ecda_0",
        type=InternLM,
        model_type="LLAMA",
        path="/mnt/petrelfs/share_data/yanhang/issue/20230921/f61851e206f4191ba8a72b3f3dac45f17de4ecda/ckpts/2710/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        module_path="/mnt/petrelfs/share_data/yanhang/issue/20230921/tf32/",
        model_config="/mnt/petrelfs/share_data/yanhang/issue/20230921/f61851e206f4191ba8a72b3f3dac45f17de4ecda/configs/Newton_20B_f61851e206f4191ba8a72b3f3dac45f17de4ecda.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),

    dict(
        abbr="a550b8122c1666e19a4a7a54991d056fe7c3e57e_0",
        type=InternLM,
        model_type="LLAMA",
        path="/mnt/petrelfs/share_data/yanhang/issue/20230921/a550b8122c1666e19a4a7a54991d056fe7c3e57e/ckpts/2710/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        module_path="/mnt/petrelfs/share_data/yanhang/issue/20230921/tf32/",
        model_config="/mnt/petrelfs/share_data/yanhang/issue/20230921/a550b8122c1666e19a4a7a54991d056fe7c3e57e/configs/Newton_20B_a550b8122c1666e19a4a7a54991d056fe7c3e57e.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),


]

# for item in range(0, 13):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_sft_v6_{100 + item*100}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v6/{100 + item*100}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v4.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)

# for item in range(0, 13):
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
