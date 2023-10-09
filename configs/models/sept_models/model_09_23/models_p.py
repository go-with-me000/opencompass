from opencompass.models import HuggingFaceCausalLM
from opencompass.models.internal import  InternLM

models = [
    dict(
        abbr="baichuan2_7b_base_00220",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_00220B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_00440",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_00440B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_00660",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_00660B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_00880",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_00880B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_01100",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_01100B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_01320",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_01320B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_01540",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_01540B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_01760",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_01760B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_01980",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_01980B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_02200",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_02200B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
    dict(
        abbr="baichuan2_7b_base_02420",
        type=InternLM,
        model_type="BAICHUAN2",
        path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_02420B/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
        tokenizer_type='baichuan2',
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),

]
