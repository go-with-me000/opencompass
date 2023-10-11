from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets.subject_story import StoryDataset

story_reader_cfg = dict(
    input_columns="prompt", output_column='answer')

story_all_sets = [
    "easy","middle","hard"
]
story_datasets = []
for _name in story_all_sets:
    story_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt="{prompt}"),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=300),
    )

    story_eval_cfg = dict(
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
                                prompt="{prompt} {prediction}")]))),
        pred_role="BOT",
    )

    story_datasets.append(
        dict(
            abbr=f"{_name}",
            type=StoryDataset,
            path="./data/subject_story",
            name=_name,
            reader_cfg=story_reader_cfg,
            infer_cfg=story_infer_cfg,
            eval_cfg=story_eval_cfg,
        ))

del _name
