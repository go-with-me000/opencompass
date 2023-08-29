from mmengine.config import read_base

with read_base():

    from ..LEvalCoursera.LEval_coursera_gen_5c84a9 import LEval_coursera_datasets
    from ..LEvalGSM100.LEval_gsm100_gen_a4d1f8 import LEval_gsm100_datasets
    from ..LEvalQuality.LEval_quality_gen_bd35f4 import LEval_quality_datasets
    from ..LEvalTopicRetrieval.LEval_topic_retrieval_gen_af0562 import LEval_tr_datasets
    from ..LEvalTPO.LEval_tpo_gen_bd35f4 import LEval_tpo_datasets

    # from ..LEvalFinancialQA.LEval_financialqa_gen_9f5404 import LEval_financialqa_datasets
    # # from ..LEvalLegalContractQA.LEval_legalcontractqa_gen_f0bb20 import LEval_legalqa_datasets
    # from ..LEvalMultidocQA.LEval_multidocqa_gen_87dc85 import LEval_multidocqa_datasets
    # from ..LEvalNaturalQuestion.LEval_naturalquestion_gen_9fec98 import LEval_nq_datasets 
    # # from ..LEvalNarrativeQA.LEval_narrativeqa_gen_9fec98 import LEval_narrativeqa_datasets
    # from ..LEvalScientificQA .LEval_scientificqa_gen_0c6e71 import LEval_scientificqa_datasets 
    
    # from ..LEvalReviewSumm.LEval_review_summ_gen_6c03d0 import LEval_review_summ_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
