from mmengine.config import read_base

with read_base():
    # from ..datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets
    # from ..datasets.nq.nq_gen_c788f6 import nq_datasets
    from ..datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ..datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    # from ..datasets.calculate.calculate_gen import calculate_datasets
    # from ..datasets.math.math_gen_265cce import math_datasets
    # from ..datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from ..datasets.tinystories.tinystories_gen import tinystories_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
