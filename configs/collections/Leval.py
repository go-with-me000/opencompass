from mmengine.config import read_base

with read_base():
    from ..datasets.LEvalCoursera.LEval_coursera_gen import LEval_coursera_datasets
    from ..datasets.LEvalFinancialQA.LEval_financialqa_gen import LEval_financialqa_datasets
    from ..datasets.LEvalGovReportSumm.LEval_gov_report_summ_gen import LEval_govreport_summ_datasets
    from ..datasets.LEvalGSM100.LEval_gsm100_gen import LEval_gsm100_datasets
    from ..datasets.LEvalLegalContractQA.LEval_legalcontractqa_gen import LEval_legalqa_datasets
    from ..datasets.LEvalMeetingSumm.LEval_meetingsumm_gen import LEval_meetingsumm_datasets
    from ..datasets.LEvalMultidocQA.LEval_multidocqa_gen import LEval_multidocqa_datasets
    from ..datasets.LEvalNarrativeQA.LEval_narrativeqa_gen import LEval_narrativeqa_datasets
    from ..datasets.LEvalNewsSumm.LEval_newssumm_gen import LEval_newssumm_datasets
    from ..datasets.LEvalPaperAssistant.LEval_paper_assistant_gen import LEval_ps_summ_datasets
    from ..datasets.LEvalPatentSumm.LEval_patent_summ_gen import LEval_patent_summ_datasets
    from ..datasets.LEvalQuality.LEval_quality_gen import LEval_quality_datasets
    from ..datasets.LEvalReviewSumm.LEval_review_summ_gen import LEval_review_summ_datasets
    from ..datasets.LEvalScientificQA.LEval_scientificqa_gen import LEval_scientificqa_datasets
    from ..datasets.LEvalTopicRetrieval.LEval_topic_retrieval_gen import LEval_tr_datasets
    from ..datasets.LEvalTPO.LEval_tpo_gen import LEval_tpo_datasets
    from ..datasets.LEvalTVShowSumm.LEval_tvshow_summ_gen import LEval_tvshow_summ_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])