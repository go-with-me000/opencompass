from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import VOLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .collections.base_small_pre import datasets
    # from .collections.C_plus import datasets
    # from .collections.example import datasets
    from .collections.base_small import datasets

    from .models.model_10_08.models_volcano import models
    # from .models.huggingface import models
    # from .models.llama import models

    from .summarizers.small import summarizer

work_dir = './outputs/2023_10_08/'

volcano_cfg = dict(
    bashrc_path="/fs-computility/llm/chenkeyu1/.bashrc",
    conda_env_name='flash2.0',
    conda_path="/fs-computility/llm/chenkeyu1/miniconda3/bin/activate",
    volcano_config_path="/fs-computility/llm/chenkeyu1/program/opencompass/configs/configs/volc_config/volcano.yaml"
)

volcano_eval_cfg = dict(
    bashrc_path="/fs-computility/llm/chenkeyu1/.bashrc",
    conda_env_name='flash2.0',
    conda_path="/fs-computility/llm/chenkeyu1/miniconda3/bin/activate",
    volcano_config_path="/fs-computility/llm/chenkeyu1/program/opencompass/configs/configs/volc_config/volcano_eval.yaml"
)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=50000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=VOLCRunner,
        volcano_cfg=volcano_cfg,
        # type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=2
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=VOLCRunner,
        volcano_cfg=volcano_eval_cfg,
        # type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        retry=2
    ),
)

# python run.py configs/eval_v.py  -r -l --debug 2>&1 | tee log.txt
