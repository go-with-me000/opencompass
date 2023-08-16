from mmengine.config import read_base

with read_base():
    from ..datasets.piqa.piqa_gen_evaluation import piqa_datasets
    from ..datasets.hellaswag.hellaswag_gen_evaluation import hellaswag_datasets
    from ..datasets.commonsenseqa.commonsenseqa_gen_evaluation import commonsenseqa_datasets
    from ..datasets.race.race_gen_evaluation import race_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
