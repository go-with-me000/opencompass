from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .collections.C_plus import datasets
    # from .collections.evaluation_gen import datasets
    from .collections.evaluation import datasets
    from .models.intern_model import models

    # from .summarizers.small import summarizer


work_dir = './outputs/evaluation/08_17/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=10000, gen_task_coef=10),
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
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        retry=10),
)

# python run.py configs/eval_evaluation.py -p llm -r -l --debug 2>&1 | tee log.txt