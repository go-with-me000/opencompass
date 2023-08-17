from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner, DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .collections.C_plus import datasets


    # from .models.llama import models
    from .models.model_08_16.models import models


work_dir = './outputs/2023_08_16/'



alillm_workspace_id = "ws1g6rf9gujt5w96"
alillm2_workspace_id = "ws1so95hgb5kn6ja"

alillm2_cfg = dict(
    bashrc_path="/cpfs01/user/chenkeyu1/.bashrc",
    conda_env_name='flash2.0',
    dlc_config_path="/cpfs01/user/chenkeyu1/.dlc/config",
    workspace_id='ws1so95hgb5kn6ja',
    worker_image='pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/chenxun-st:llm-test',
)
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=alillm2_cfg,
        max_num_workers=2,
        task=dict(type=OpenICLInferTask),
        retry=2),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        # aliyun_cfg=alillm2_cfg,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
),
)

# python run.py configs/eval_aliyun.py -p llm -r -l --debug 2>&1 | tee log.txt