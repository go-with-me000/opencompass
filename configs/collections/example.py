from mmengine.config import read_base

with read_base():
    # from ..datasets.FewCLUE_bustm.FewCLUE_bustm_ppl_9ef540 import bustm_datasets
    from ..datasets.nq.nq_gen_c788f6 import nq_datasets
    # from ..datasets.mmlu.mmlu_gen_test import mmlu_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
