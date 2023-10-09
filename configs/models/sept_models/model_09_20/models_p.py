from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="En-1b-best_27000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/sft/en-1b/27000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/",
    #     model_config="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/en-1b.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="En-1b-std_40000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/sft/en-1b-resume/40000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/",
    #     model_config="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/en-1b-resume.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="En-1b-attn_73000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/sft/en-1b-init-pre/73000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/",
    #     model_config="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/en-1b-init-pre.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="En-1b-ffn_58000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/sft/en-1b-init-param/58000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/",
    #     model_config="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/en-1b-init-param.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    #
    # dict(
    #     abbr="En-1b-base_60000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/sft/en-1b-probe/60000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/",
    #     model_config="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/en-1b-base.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    #
    # dict(
    #     abbr="En-1b-Norm_69000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/sft/en-1b-probe-norm/69000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_en/",
    #     model_config="/mnt/petrelfs/share_data/zhouyunhua/train_internlm_norm/configs/en-1b-probe-norm.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     abbr="mammoth_llama2_7b_200",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0918/200",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_zzy_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="mammoth_llama2_7b_400",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0918/400",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_zzy_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="mammoth_llama2_7b_600",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0918/600",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_zzy_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="mammoth_llama2_7b_1000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0918/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_zzy_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

]

# for item in range(0, 19):
#     model_info = dict(
#         abbr=f"7B_llama2_coder_{64000+item*2000}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_v3/{64000+item*2000}",
#         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
#         tokenizer_type='llama',
#         module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
#         model_config="/mnt/petrelfs/chenkeyu1/train_internlm/configs/7B_llama_coder.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1)
#     )
#     models.append(model_info)

for item in range(0, 13):
    model_info = dict(
        abbr=f"7B_llama_coder_sft_v2_{100+item*100}",
        type=InternLM,
        path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v2/{100+item*100}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v2.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)

for item in range(0, 13):
    model_info = dict(
        abbr=f"7B_llama_coder_sft_v3_{100+item*100}",
        type=InternLM,
        path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_sft_v3/{100+item*100}",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/7B_llama2_coder_sft_v3.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
    models.append(model_info)

for item in range(0, 12):
    model_info = dict(
        abbr=f"13B_codellama_sft_{100+item*100}",
        type=InternLM,
        path=f"s3://checkpoints_ssd_02/songdemin/13B_codellama_sft/{100+item*100}",
        tokenizer_path='/mnt/petrelfs/share_data/llm_llama/codellama_raw/codellama_tokenizer.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share/songdemin/code/train_internlm/",
        model_config="/mnt/petrelfs/share/songdemin/code/train_internlm/configs/13B_codellama_sft.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=2, num_procs=2)
    )
    models.append(model_info)
