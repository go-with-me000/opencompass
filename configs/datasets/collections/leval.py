from mmengine.config import read_base

with read_base():
    from ..gsm8k.gsm8k_gen_collie import gsm8k_datasets
    from ..humaneval.humaneval_gen_collie import humaneval_datasets
    from ..mmlu.mmlu_gen_collie import mmlu_datasets
    from ..bbh.bbh_gen_collie import bbh_datasets
    from ..alpaca_farm.alpaca_farm_gen_collie import alpaca_farm_datasets

  

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
