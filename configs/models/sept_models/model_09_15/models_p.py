from opencompass.models.internal import  InternLM

models = [
    # dict(
    #     abbr="copernicus_7b_v0_5_1_2000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/copernicus_7b_v0.5.1/2000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.5.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="copernicus_7b_v0_5_1_3000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/copernicus_7b_v0.5.1/3000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.5.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="copernicus_7b_v0_5_1_4000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/copernicus_7b_v0.5.1/4000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.5.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    #
    # dict(
    #     abbr="copernicus_7b_v0_6_1_1000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/copernicus_7b_v0.6.1/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/",
    #     model_config="/mnt/petrelfs/feizhaoye.dispatch/train_internlm_2/configs/copernicus_7b_v0.6.1.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     abbr="llama_7B_calc_yf_0915_400",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_calc_yf_0915/400",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_calc_yf_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     abbr="LLAMA-MTP-7B_40000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/pengrunyu/demo_7B_mtp/40000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/pengrunyu/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/pengrunyu/train_internlm/configs/demo_7B_mtp.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr="LLAMA-MTP-7B-EN_21000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/pengrunyu/demo_7B_mtp_en/21000/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/pengrunyu/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/pengrunyu/train_internlm/configs/demo_7B_mtp.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     abbr="wizard_math_zzy_200",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0915/200",
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
    #     abbr="wizard_math_zzy_400",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0915/400",
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
    #     abbr="wizard_math_zzy_600",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0915/600",
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
    #     abbr="wizard_math_zzy_1000",
    #     type=InternLM,
    #     model_type="LLAMA",
    #     path="s3://checkpoints_ssd_02/further_llama_7B_math_zzy_0915/1000",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/shaoyunfan/train_internlm/configs/further_llama_7B_math_zzy_eval.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr="LLAMA-MTP-7B-_50000",
        type=InternLM,
        model_type="LLAMA",
        path="s3://checkpoints_ssd_02/pengrunyu/demo_7B_mtp/50000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="/mnt/petrelfs/share_data/pengrunyu/train_internlm/",
        model_config="/mnt/petrelfs/share_data/pengrunyu/train_internlm/configs/demo_7B_mtp.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]

# for item in range(0, 7):
#     model_info = dict(
#         abbr=f"7B_llama_coder_{50000+item*2000}",
#         type=InternLM,
#         path=f"s3://checkpoints_ssd_02/songdemin/7B_llama2_coder_v3/{50000+item*2000}",
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
