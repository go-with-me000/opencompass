from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .collections.C_plus import datasets
    # from .collections.example import datasets
    from .collections.base_small import datasets


    # from .models.llama import models
    from .models.model_08_18.models_p import models
    # from .models.huggingface import models



work_dir = './outputs/2023_08_18/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=5),
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
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        retry=4),
)

# python run.py configs/eval_p.py -p llm-p2 -r -l --debug 2>&1 | tee log.txt