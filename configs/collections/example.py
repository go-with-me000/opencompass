from mmengine.config import read_base

with read_base():
    from ..datasets.piqa.piqa_gen_1194eb import piqa_datasets
    # from ..datasets.nq.nq_gen_c788f6 import nq_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
