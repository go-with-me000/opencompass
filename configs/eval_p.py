from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
<<<<<<< HEAD
    from .datasets.collections.C import datasets
    # from .collections.evaluation import datasets
    # from .datasets.collections.example import datasets
    from .my_model.modelv1 import models

    # from .summarizers.small import summarizer


work_dir = './outputs/2023_08_14/'
=======
    # from .collections.C_plus import datasets
    # from .collections.example import datasets
    from .collections.base_small import datasets


    # from .models.llama import models
    from .models.model_08_18.models_p import models
    # from .models.huggingface import models



work_dir = './outputs/2023_08_20/'
>>>>>>> cky/0.6.1

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=10000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
<<<<<<< HEAD
        max_num_workers=32,
        task=dict(type=OpenICLInferTask),
        retry=2),
=======
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
        retry=4),
>>>>>>> cky/0.6.1
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
<<<<<<< HEAD
        max_num_workers=5,
        task=dict(type=OpenICLEvalTask),
        retry=10),
)

# python run.py configs/eval_p.py -p llm-p2 -r -l --debug 2>&1 | tee log.txt
=======
        max_num_workers=100,
        task=dict(type=OpenICLEvalTask),
        retry=4),
)

# python run.py configs/eval_p.py -p p4_test -r -l --debug 2>&1 | tee log.txt
>>>>>>> cky/0.6.1
