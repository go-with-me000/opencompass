# OpenCompass

## Installation

1. Prepare Torch refer to [PyTorch](https://pytorch.org/).

Notice that OpenCompass requires `pytorch>=1.13`.

```bash
conda create --name opencompass python=3.8 -y
conda activate opencompass
conda install pytorch torchvision -c pytorch
```

2. Install OpenCompass:

```bash
git clone https://github.com/opencompass/opencompass
cd opencompass
pip install -r requirements/runtime.txt
pip install -e .
```

3. Install Llama (option)

install llama if you want to eval llama models.

```
git clone https://github.com/facebookresearch/llama
pip install -r requirements.txt
pip install 'flash_attn<1.0.0'
pip install -e llama
```

Modify `llama/model.py` line 237 into `output = self.output(h[:, :, :])`

if install `flash_attn` failed. try intsall it from score:

<details><summary>click to show the detail</summary>

```bash
git clone -b v0.2.8 https://github.com/HazyResearch/flash-attention.git
pip install -r requirements.txt
pip install -e .

# if you encount error:'No module named 'dropout_layer_norm' or the like
# complie ops
cd csrc/layer_norm && python setup.py install
cd csrc/rotary && python setup.py install
cd csrc/fused_dense_lib && python setup.py install
```

</details>

4. Install humaneval (option)

do this if you want to eval on humaneval dataset.

```
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -r requirements.txt
pip install -e .
```

Remember to remove the comments and uncomment [this line](https://github.com/openai/human-eval/blob/312c5e5532f0e0470bf47f77a6243e02a61da530/human_eval/execution.py#L58) in the source code.

## Usage

配置好 PetrelFS (`~/petreloss.conf`):

```text
[opennlplab_hdd]

enable_mc = False
boto = True
host_base = http://10.140.2.204:80
access_key = YOUR_ACCESS_KEY
secret_key = YOUR_SECRET_KEY
```

将数据集下载或软链到 `./data` 处

```bash
ln -s /mnt/petrelfs/share_data/zhoufengzhe/llm_evaluation ./data
# 或者从 gitlab 上下载
# git clone ssh://git@gitlab.pjlab.org.cn:1122/openmmlab/bigmodel/llm-evaluation-datasets.git ./data
```

最后运行命令进行全量测试。例如：

```bash
python run.py configs/eval_demo.py -p llm_it
```

在后续的运行中，若能保证 huggingface 相关的代码均已被正确缓存，可使用以下环境变量，在不联网的机器上运行

```bash
export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1;
```

## 快速上手

本工具基于 MMEngine 的配置文件，可以快速部署评估任务。

程序入口为 `run.py`，使用方法如下：

```bash
run.py [-p PARTITION] [-q QUOTATYPE] [--debug] [-m MODE] [-r [REUSE]] [-w WORKDIR] config
```

参数解释如下：

- `-p` 指定 slurm 分区；
- `-q` 指定 slurm quotatype （默认为 auto），可选 reserved, auto, spot；
- `--debug` 开启时，推理和评测任务会以单进程模式运行，且输出会实时回显，便于调试；
- `-m` 运行模式，默认为 `all`。可以指定为 `infer` 则仅运行推理，获得输出结果；如果在 `{WORKDIR}` 中已经有模型输出，则指定为 `eval` 仅运行评测，获得评测结果；如果在 `results` 中已有单项评测结果，则指定为 `viz` 仅运行可视化；指定为 `all` 则同时运行推理和评测。
- `-r` 重用已有的推理结果。如果后面跟有时间戳，则会复用工作路径下该时间戳的结果；否则则复用指定工作路径下的最新结果。
- `-w` 指定工作路径，默认为 `./outputs/default`
- `-l` 打开飞书机器人状态上报。[详细文档](docs/zh_cn/tools.md#lark-bot)

整体运行流如下：

1. 读取配置文件，解析出模型、数据集、评估器等配置信息
2. 评测任务主要分为推理 `infer`、评测 `eval` 和可视化 `viz` 三个阶段，其中推理和评测经过 `Partitioner` 进行任务切分后，交由 `Runner` 负责并行执行。单个推理和评测任务则被抽象成 `OpenICLInferTask` 和 `OpenICLEvalTask`。
3. 两阶段分别结束后，可视化阶段会读取 `results` 中的评测结果，生成可视化报告。

实际运行时，你可以通过制定 -m 来指定运行模式。

## 实用工具

见 [实用工具](docs/zh_cn/tools.md)。

## 开发规范

见 [文档](https://aicarrier.feishu.cn/wiki/wikcnocfGDlTixegjAgstKP476e)

### 配置解释

#### `infer`, `eval`

分别配置推理和评测的配置，支持的参数如下：

```python
infer = dict(
    partitioner=dict(type='SizePartitioner', max_task_size=2000),  # 任务切分策略，支持 SizePartitioner 和 NaivePartitioner。对于 SizePartitioner，可以使用 max_task_size 指定单个任务的最大长度，超过该长度的任务会被切分成多个子任务。
    runner=dict(
        type='SlurmRunner',  # 运行器，支持 SlurmRunner, LocalRunner 和 DLCRunner
        max_num_workers=2,  # 最大并行评测任务数，建议不要太离谱
        task=dict(type='OpenICLInferTask'),
        retry=5),  # 任务失败后重试次数，防止如端口被占用等情况 block 评测
)
```

阿里云的 `DLCRunner` 配置可以参考 `configs/aliyun_intern_benchmark_chat.py`。

<!-- #### `evaluator`

支持主观评测和客观评测的 Evaluator。目前，`ICLEvaluator` 用于客观评测，通过调用 OpenICL 得到评测结果。

```python
# max_num_workers 指开启多少个进程来进行评测
evaluator = dict(type='ICLEvaluator', max_num_workers=16)
``` -->

#### Model

`models` 是一个列表，每个元素是一个字典，包含了模型的配置信息。

```python
models = [
    dict(type='LLM',  # Model Wrapper 类别， 支持 "LLM", "LLama", "OpenAI", 通常使用 LLM
         path='/mnt/petrelfs/share_data/gaotong/llm/sft_7132k_moss/13999',  # 模型路径, 同样支持 ceph
         max_out_len=100,  # 模型在生成任务中最大输出长度
         max_seq_len=2048,  # 模型整体最大处理长度 （包括 prompt + 输出）
         batch_size=16,  # 模型推理的 batch size
         run_cfg=dict(num_gpus=2)),  # 模型推理所需的 GPU 数量
]
```

此外，从 0.3.0 版本起，我们支持了与模型绑定的 meta prompt。[文档](docs/zh_cn/meta_prompt.md)

#### Dataset

`datasets` 是一个列表，每个元素是一个字典，包含了数据集的配置信息。
为了便于管理，数据集的配置被放置于 `configs/datasets` 目录下，在主 config 文件中通过继承的方式引入。
例如，我们需要引入 piqa 的数据集配置，可以在 `configs/eval.py` 中添加如下代码：

```python
_base_ = [
    'datasets/piqa.py',
]
datasets = []
datasets += _base_.piqa_datasets
```

约定 `_base_` 中的数据集配置变量名为 `{DATASET_NAME}_datasets`。以 piqa 为例，其数据集配置为：

```python
piqa_datasets = [
    dict(type='HFDataset',  # 数据集类型，影响数据集加载方式。HFDataset 是基于 huggingface 的数据集加载方式，其他可选项见 opencompass/load_datasets.py
         abbr='piqa', # 数据集简称，用于生成数据集路径
         path='piqa', # 数据集路径，为 datasets.load_dataset 的必须参数
         infer_cfg=piqa_infer_cfg, # 推理配置，见下文
         eval_cfg=piqa_eval_cfg)  # 评估配置，见下文
]
```

其中，`abbr`、`infer_cfg` 和 `eval_cfg` 是保留字段，他们用于指定模型输出的文件名、推理配置和评估配置；其他参数将会被传到 opencompass/load_datasets.py 中相应的数据集加载函数中。

`eval_cfg`:

指定了客观评测时的评估配置，包含了评估器的类型和数据集的划分。

```python

eval_cfg = dict(evaluator=dict(type='AccEvaluator'),  # OpenICL 中的评估器类型，可用类型见 opencompass/openicl/openicl/icl_evaluator/__init__.py
                    ds_split='validation',  # 用于评测的数据集 split
                    ds_column='label',  # 数据集中的标签列
                    dataset_postprocessor=dict(type='general'),  # 数据集内容（str）的处理函数，评估分数前会先运行该函数
                    pred_postprocessor=dict(type='general'),  # 预测结果（str）的处理函数，评估分数前会先运行该函数
                    )
```

`infer_cfg`:

这里指定的是推理配置，包含了数据集读取、ICE 模板 (k-shot 模板)、prompt 模板，检索器和推理器的配置。这里直接包裹的是 OpenICL 的推理流程，具体参数含义不再赘述。

```python
piqa_infer_cfg = dict(reader=dict(type='DatasetReader',
                                  input_columns=['goal', 'sol1', 'sol2'],
                                  output_column='label'),
                      ice_template=dict(type='PromptTemplate',
                                        template={
                                            0: '</E>Q: </G>\nA: </S1>\n',
                                            1: '</E>Q: </G>\nA: </S2>\n'
                                        },
                                        column_token_map={
                                            'sol1': '</S1>',
                                            'sol2': '</S2>',
                                            'goal': '</G>'
                                        },
                                        ice_token='</E>'),
                      prompt_template=dict(type='PromptTemplate',
                                        template={
                                            0: '</E>Q: </G>\nA: </S1>\n',
                                            1: '</E>Q: </G>\nA: </S2>\n'
                                        },
                                        column_token_map={
                                            'sol1': '</S1>',
                                            'sol2': '</S2>',
                                            'goal': '</G>'
                                        },
                                        ice_token='</E>'),
                      retriever=dict(type='ZeroRetriever',
                                     test_split='validation'),
                      inferencer=dict(type='PPLInferencer'))
```

## FAQ

1. 出现与 NCCL 相关的问题，初步可以尝试在运行的启动命令之前添加: `NCCL_SOCKET_IFNAME=eth0`

## Acknowledgement

This repository borrows part of the code from the [OpenICL](https://github.com/Shark-NLP/OpenICL) repository.
