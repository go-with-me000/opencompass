from mmengine.config import read_base

with read_base():
    # from ..datasets.tinystories.tinystories_gen import tinystories_datasets
    # from ..datasets.subjectivity.subjectivity import subjectivity_datasets
    from ..datasets.subjectivity.story import story_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
