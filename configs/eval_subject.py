from mmengine.config import read_base

from opencompass.models import OpenAI
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .collections.example import datasets
    from .collections.subjectivity_only import datasets
    from .models.huggingface import models


work_dir = './outputs/2023_10_10/'

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=2000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
        retry=3),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLEvalTask,
                  judge_cfg=dict(abbr='GPT4.0',
                                 type=OpenAI, path='gpt-4',
                                 key="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",
                                 meta_template=api_meta_template,
                                 query_per_second=1,
                                 max_out_len=2048, max_seq_len=2048, batch_size=4)
                  )),
)

# python run.py configs/eval_subject.py -p llm_t -r -l --debug 2>&1 | tee log.txt