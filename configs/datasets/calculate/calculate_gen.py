from opencompass.openicl import AccEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import  CalculateDataset, calculate_postprocess, CalculateDataset_V2

calculate_reader_cfg = dict(
    input_columns=['question'], output_column='answer')

calculate_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="当前公式:{question}\n"
#         template='''当前公式:(0.9905-(-334))
# 回答：0.9905+334 小数点位数=4 计算步骤 0009905 + 3340000 A=5, C=0 A=0, C=0 A=9, C=0 A=9, C=0 A=4, C=0 A=3, C=0 A=3, C=0 计算中间结果 3349905 小数点位数=4 计算结果 334.9905
# 答案是 334.9905
#
# 请计算下列公式:(437*(-36.8301))
# 回答：437*-36.8301 计算步骤:积为- 小数点位数=5 计算步骤 4370 * 368301 计算步骤 4370*1=4370 计算步骤 4370*0=00000 计算步骤 4370*3=1311000 计算步骤 4370*8=34960000 计算步骤 4370*6=262200000 计算步骤 4370*3=1311000000 叠加计算中间结果 1609475370 小数点位数=5 计算步骤:积为- 计算结果 -16094.7537
# 答案是 -16094.7537
#
# 请计算下列公式:(1.49-1.8033)
# 回答：1.49-1.8033 判断被减数1.8033 - 1.49, 结果为- 小数点位数=4 计算步骤 18033 - 14900 D=3, B=0 D=3, B=0 D=1, B=1 D=3, B=0 D=0, B=0 计算中间结果 3133 小数点位数=4 计算结果-0.3133
# 答案是 -0.3133
#
# 请计算下列公式:(1.152-670)
# 回答：1.152-670 判断被减数670.0 - 1.152, 结果为- 小数点位数=3 计算步骤 670000 - 001152 D=8, B=1 D=4, B=1 D=8, B=1 D=8, B=1 D=6, B=0 D=6, B=0 计算中间结果 668848 小数点位数=3 计算结果-668.848
# 答案是 -668.848
#
# 请计算下列公式:{question}
# 回答：
# '''
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=3000))

calculate_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                          pred_postprocessor=dict(type=calculate_postprocess),)

calculate_datasets = [
    dict(
        type=CalculateDataset_V2,
        abbr='calculate',
        path='./data/calculate/calculate_test_v2.jsonl',
        reader_cfg=calculate_reader_cfg,
        infer_cfg=calculate_infer_cfg,
        eval_cfg=calculate_eval_cfg)
]
