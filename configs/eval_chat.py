from mmengine.config import read_base
from opencompass.models.internal import InternLMwithModule, InternLM
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from copy import deepcopy
import os.path as osp

with read_base():
    from .collections.chat_medium2 import datasets

    # from .models.model_10_07.models_p import models

    from .summarizers.medium_report import summarizer

# datasets 在 from ..datasets.collections.chat_medium import datasets 已经设置好了
# datasets = [..]
work_dir = './outputs/2023_10_08/'


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
    # dict(
    #     abbr="sft_7b_t_5260",
    #     type=InternLM,
    #     model_type='INTERNLM',
    #     path="/mnt/petrelfs/share_data/zhangwenwei/models/llmit/exps/sft_7b_t/5260",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path="/mnt/petrelfs/chenkeyu1/files/train_internlm/",
    #     model_config="/mnt/petrelfs/llmit/code/opencompass_v051/train_internlm/configs/maibao_7b_8k_sft.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    #     meta_template=meta_template,
    # ),
    dict(
        abbr="e11a714_Newton_2795",
        type=InternLM,
        model_type="LLAMA",
        path="/mnt/petrelfs/share_data/gaoyang/ckpts/e11a714_Newton_20B_0.2.0@6500_toolbench_safety/2795",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        module_path="/mnt/petrelfs/share_data/yanhang/issue/20230921/tf32/",
        model_config="/mnt/petrelfs/share_data/yanhang/issue/20230921/a550b8122c1666e19a4a7a54991d056fe7c3e57e/configs/Newton_20B_a550b8122c1666e19a4a7a54991d056fe7c3e57e.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),
]

#  python run.py configs/eval_chat.py -p llm_t -r --debug
