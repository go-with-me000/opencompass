from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets.subjectivity import SubjectivityDataset

subjectivity_reader_cfg = dict(
    input_columns="input", output_column='output', train_split='test')

subjectivity_all_sets = [
    "subjectivity",
]

subjectivity_datasets = []
for _name in subjectivity_all_sets:
    subjectivity_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt="{input}"),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=100),
    )

    subjectivity_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role="SYSTEM",
                            fallback_role="HUMAN",
                            prompt="请为下面句子的流畅性打分，分数为一个1-5之间的整数。除数字外无需返回其他任何内容。"
                        ),
                    ],
                    round=[dict(role="HUMAN",
                                prompt="{input} {prediction}")]))),
        pred_role="BOT",
    )

    subjectivity_datasets.append(
        dict(
            abbr=f"{_name}",
            type=SubjectivityDataset,
            path="./data/subjectivity/",
            name=_name,
            reader_cfg=subjectivity_reader_cfg,
            infer_cfg=subjectivity_infer_cfg,
            eval_cfg=subjectivity_eval_cfg,
        ))

del _name
