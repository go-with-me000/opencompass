from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator
from opencompass.datasets import LEvalNarrativeQADataset

LEval_narrativeqa_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_narrativeqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{context}\nQuestion: {question}?\nAnswer:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50)
)

LEval_narrativeqa_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator), 
    pred_role='BOT'
)

LEval_narrativeqa_datasets = [
    dict(
        type=LEvalNarrativeQADataset,
        abbr='LEval_narrativeqa',
        path='L4NLP/LEval',
        name='narrative_qa',
        reader_cfg=LEval_narrativeqa_reader_cfg,
        infer_cfg=LEval_narrativeqa_infer_cfg,
        eval_cfg=LEval_narrativeqa_eval_cfg)
]
