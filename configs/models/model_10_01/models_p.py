from opencompass.models.internal import InternLM

models = [
    # dict(
    #     abbr="Newton_2795",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="/mnt/petrelfs/share_data/gaoyang/ckpts/0.4_Newton_20B_0.4.0@6500_toolbench_safety/2795/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/share_data/yanhang/issue/20230921/tf32/",
    #     model_config="/mnt/petrelfs/share_data/yanhang/issue/20230921/a550b8122c1666e19a4a7a54991d056fe7c3e57e/configs/Newton_20B_a550b8122c1666e19a4a7a54991d056fe7c3e57e.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=4, num_procs=4)
    # ),

    # dict(
    #     abbr="1b_v1_28000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/wangbo/1B_v2/28000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="1b_v1_28000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v2/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/wangbo/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/wangbo/1b_v2/1B_v2.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

]
# for item in range(0, 5):
#     model_info = dict(
#         abbr=f"bohr_v0.1.0_{2000 + item*2000}",
#         type=InternLM,
#         path=f"/mnt/petrelfs/share_data/shaoyunfan/ckpt/bohr_0927/{2000 + item*2000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/hwfile/share_data/shaoyunfan/train_internlm/",
#         model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/bohr.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)


for item in range(0, 4):
    model_info = dict(
        abbr=f"baichuan_train_7b_base_v2_{10000 + item * 10000}",
        type=InternLM,
        model_type="BAICHUAN2",
        path=f"/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v2/{10000 + item * 10000}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)

for item in range(0, 6):
    if item == 5:
        model_info = dict(
            abbr=f"baichuan_train_7b_base_v3_52000",
            type=InternLM,
            model_type="BAICHUAN2",
            path=f"/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v3/52000",
            tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
            tokenizer_type='baichuan2',
            module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
            model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
            max_out_len=100,
            max_seq_len=2048,
            batch_size=16,
            run_cfg=dict(num_gpus=1, num_procs=1)
        )
        models.append(model_info)
    else:
        model_info = dict(
            abbr=f"baichuan_train_7b_base_v3_{10000 + item * 10000}",
            type=InternLM,
            model_type="BAICHUAN2",
            path=f"/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v3/{10000 + item * 10000}",
            tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
            tokenizer_type='baichuan2',
            module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
            model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
            max_out_len=100,
            max_seq_len=2048,
            batch_size=16,
            run_cfg=dict(num_gpus=1, num_procs=1)
        )
        models.append(model_info)

for item in range(0, 4):
    model_info = dict(
        abbr=f"baichuan_train_7b_base_v4_{10000 + item * 10000}",
        type=InternLM,
        model_type="BAICHUAN2",
        path=f"/mnt/petrelfs/share_data/yangxiaogui/model_ckpt/baichuan2_further_train/baichuan_train_7b_base_v4/{10000 + item * 10000}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)
