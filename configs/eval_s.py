from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .datasets.collections.C import datasets
    from .datasets.collections.base_small import datasets
    from .models.my_model import models

    # from .summarizers.small import summarizer


work_dir = './outputs/2023_08_02/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=15),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=4),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        retry=4),
)