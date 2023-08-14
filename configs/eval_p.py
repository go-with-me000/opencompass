from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.collections.C import datasets
    # from .collections.evaluation import datasets
    from .my_model.modelv1 import models

    # from .summarizers.small import summarizer


work_dir = './outputs/2023_08_11/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=32,
        task=dict(type=OpenICLInferTask),
        retry=2),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=5,
        task=dict(type=OpenICLEvalTask),
        retry=10),
)

# python run.py configs/eval_p.py -p llm-p2 -r -l --debug 2>&1 | tee log.txt