from mmengine.config import read_base
from opencompass.models.internal import InternLMwithModule, InternLM
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from copy import deepcopy
import os.path as osp

with read_base():
    from .collections.chat_medium2 import datasets
    from .summarizers.medium_report import summarizer

# datasets 在 from ..datasets.collections.chat_medium import datasets 已经设置好了
# datasets = [..]
work_dir = './outputs/2023_09_27/'


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=3),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        retry=4),
)


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<TOKENS_UNUSED_1>\n', generate=True),
    ],
    eos_token_id=103028)
models = [
    dict(
        abbr="f61851e206f4191ba8a72b3f3dac45f17de4ecda_1",
        type=InternLM,
        model_type="LLAMA",
        path="/mnt/petrelfs/share_data/yanhang/issue/20230921/f61851e206f4191ba8a72b3f3dac45f17de4ecda/ckpts/2710/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        meta_template=meta_template,
        module_path="/mnt/petrelfs/share_data/yanhang/issue/20230921/tf32/",
        model_config="/mnt/petrelfs/share_data/yanhang/issue/20230921/f61851e206f4191ba8a72b3f3dac45f17de4ecda/configs/Newton_20B_f61851e206f4191ba8a72b3f3dac45f17de4ecda.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),

    dict(
        abbr="a550b8122c1666e19a4a7a54991d056fe7c3e57e_1",
        type=InternLM,
        model_type="LLAMA",
        path="/mnt/petrelfs/share_data/yanhang/issue/20230921/a550b8122c1666e19a4a7a54991d056fe7c3e57e/ckpts/2710/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        meta_template=meta_template,
        module_path="/mnt/petrelfs/share_data/yanhang/issue/20230921/tf32/",
        model_config="/mnt/petrelfs/share_data/yanhang/issue/20230921/a550b8122c1666e19a4a7a54991d056fe7c3e57e/configs/Newton_20B_a550b8122c1666e19a4a7a54991d056fe7c3e57e.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),

]

#  python run.py configs/eval_chat.py -p llm_t -r --debug
