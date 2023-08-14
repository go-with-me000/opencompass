from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .datasets.piqa.piqa_ppl import piqa_datasets
    # from .datasets.siqa.siqa_gen import siqa_datasets
    # from .datasets.nq.nq_gen_c788f6 import nq_datasets
    from .datasets.collections.C import datasets
    # from .models.hf_llama2_7b import models
    from .models.internlm_7b import models
    # from .summarizers.medium import summarizer



work_dir = './outputs/2023_08_01/'
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=5000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        ),
)