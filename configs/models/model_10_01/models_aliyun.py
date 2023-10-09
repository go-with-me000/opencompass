from opencompass.models.internal import InternLM, LLMv3, LLama

models = [
    # dict(
    #     abbr="Euclid_70B_0.1.0_41000",
    #     type=InternLM,
    #     path="/cpfs01/shared/alillm2/alillm2_hdd/zhangshuo/ckpts/Euclid_70B_0.1.0/41000",
    #     tokenizer_path='/cpfs01/shared/public/tokenizers/llama.model',
    #     tokenizer_type='llama',
    #     module_path="/cpfs01/shared/public/zhangshuo/train_internlm/",
    #     model_config="/cpfs01/shared/public/zhangshuo/train_internlm/configs/Euclid_70B_0.1.0.py",
    #     model_type="LLAMA",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=8, num_procs=8)
    # )

dict(abbr="LLama2-70B",
         type=LLama, path='/cpfs01/shared/public/zhangshuo/ckpts/llama-2-70b/',
         tokenizer_path='/cpfs01/shared/public/tokenizers/llama.model',
         tokenizer_type='llama',
         max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8)),
]
