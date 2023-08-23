from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.collections.C_fepe import datasets
    from .my_model.liuxiaoran import models

work_dir = './outputs/2023_08_21/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
        retry=4),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask),
        retry=4),
)

# python run.py configs/eval_lxr.py -p llm --debug 调试用
# python run.py configs/eval_lxr.py -p llm 第一次用
# python run.py configs/eval_lxr.py -p llm -r 第二次用
