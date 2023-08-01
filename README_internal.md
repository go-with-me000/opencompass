# OpenCompass 内部版极速上手

## 安装

1. 准备 OpenCompass 运行环境并安装：

   ```bash
   conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
   conda activate opencompass
   git clone ssh://git@gitlab.pjlab.org.cn:1122/openmmlab/bigmodel/opencompass.git
   cd opencompass
   pip install -e .
   ```

2. 安装 humaneval（可选）：

   如果你需要**在 humaneval 数据集上评估模型代码能力**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/openai/human-eval.git
   cd human-eval
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   请仔细阅读 `human_eval/execution.py` **第48-57行**的注释，了解执行模型生成的代码可能存在的风险，如果接受这些风险，请取消**第58行**的注释，启用代码执行评测。

   </details>

3. 安装 Llama（可选）：

   如果你需要**使用官方实现评测 Llama / Llama-2 / Llama-2-chat 模型**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/facebookresearch/llama.git
   cd llama
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   你可以在 `configs/models` 下找到所有 Llama / Llama-2 / Llama-2-chat 模型的配置文件示例。([示例](https://github.com/InternLM/opencompass/blob/eb4822a94d624a4e16db03adeb7a59bbd10c2012/configs/models/llama2_7b_chat.py))

   </details>

## 配置

将数据集下载或软链到 `./data` 处

```bash
ln -s /mnt/petrelfs/share_data/zhoufengzhe/llm_evaluation ./data
# 或者从 gitlab 上下载
# git clone ssh://git@gitlab.pjlab.org.cn:1122/openmmlab/bigmodel/llm-evaluation-datasets.git ./data
```

如果需要访问 ceph 上的权重，需要联系对应分区管理员开通读取权限，并配置好 PetrelFS (`~/petreloss.conf`)。以下为示例配置:

```text
[opennlplab_hdd]
enable_mc = False
boto = True
host_base = http://10.140.2.204:80
access_key = YOUR_ACCESS_KEY
secret_key = YOUR_SECRET_KEY

[model_weights]
enable_mc = False
boto = True
host_base = http://10.140.2.254:80
access_key = YOUR_ACCESS_KEY
secret_key = YOUR_SECRET_KEY
```

在后续的运行中，若能保证 huggingface 相关的代码均已被正确缓存，可使用以下环境变量，在不联网的机器上运行

```bash
export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1;
```

## 启动评测

在 OpenCompass 中进行评测，需要首先准备一份配置文件，再通过命令启动任务。

配置文件配置项相对较多，配置起来相对繁琐。为了方便使用，**建议各团队在迭代模型时，由负责人维护一份基础配置文件，配置好所需测试的数据集及模型，再分发给有测试需要的组员。而组员在迭代时则只需要更新 `models` 字段中的 `path`，`tokenizer_path` 及 `abbr` 字段，并运行命令即可启动测试，而无需关注其它配置细节。**

本章以内部模型的测试为例，介绍如何使用 OpenCompass 进行评测。本文仅介绍关键的配置点及命令。建议**需要速成的组员**阅读。如果需要完全入门，建议参考社区版的[快速上手](docs/zh_cn/get_started.md#快速上手)。

### 准备配置文件

OpenCompass 使用 python 格式的配置文件，通常放在 `configs/` 目录下。运行时，我们只需要配置 `datasets` 及 `models` 字段，设置待测的模型及数据集。

例如，下面配置将会测试 `PJLM-v0.2.0-Exam-v0.1.5` 在 winograd 和 siqa 上的表现：

```python
from mmengine.config import read_base
from opencompass.models.internal import LLMv2

with read_base():
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

datasets = [*siqa_datasets, *winograd_datasets]

models = [
    dict(
        abbr='PJLM-v0.2.0-Exam-v0.1.5',  # 定义最后在报告中显示的模型名称
        path='model_weights:s3://model_weights/0331/1006_pr/5499/',  # 模型的路径，允许是 ceph 路径
        tokenizer_path=
        '/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',  # 模型的 tokenizer 路径
        type=LLMv2,
        model_type='converted',
        tokenizer_type='v4',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=8, num_procs=8)),  # 运行时所需的 gpu 及内存数，跑内部 LLM 时要求 num_gpus 和 num_procs 相同
]
```

更多关于配置的介绍请阅读社区版本的 [快速上手](docs/zh_cn/get_started.md#快速上手)。

### 运行测试 （S 集群）

假设上面配置文件已经保存为 `configs/eval.py`。

在 S 集群管理机上运行以下命令，即可在 `llmeval` 分区上并行启动 32 个 `quotatype` 为 `auto` 的评测任务：

```bash
python run.py configs/eval.py --slurm -p llmeval -q auto --max-num-workers 32
```

为了充分利用资源，OpenCompass 会将评测任务按计算量拆分为多个子任务，并按照用户指定的 `max-num-workers` 并行启动。每个任务资源占用取决于模型的配置，如本例，每个子任务会占用 8 个 GPU。

由于每个子任务在启动时都存在读取模型的开销，如果子任务太多，也会拖慢评测速度。因此，也建议灵活调整 `--max-partition-size` 参数，增大单个子任务的大小。（默认为2000）

```bash
python run.py configs/eval.py --slurm -p llmeval -q auto --max-num-workers 32 --max-partition-size 4000
```

另外，OpenCompass 默认使用并行的方式进行评测，不利于调试。为了便于及时发现问题，我们可以在首次启动时使用 debug 模式运行，该模式会将任务串行执行，并会实时输出任务的执行进度。

```bash
python run.py configs/eval.py --slurm -p llmeval -q auto --max-num-workers 32 --debug
```

如果一切正常，屏幕上会出现 "Starting inference process"：

```bash
[2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

此时可以使用 `ctrl+c` 中断 debug 模式的执行，重新进行并行评测。

`run.py` 还支持在本机或阿里云上启动任务。更多介绍请参考 [评测任务发起](docs/zh_cn/user_guides/experimentation.md#评测任务发起)

### 运行测试 （阿里云）

在上述配置文件中添加以下行

```python
with read_base():
    from .aliyun_env import llm_infer as infer, llm_eval as eval  # 使用不同的 workspace 需要 import 不同的配置
```

并按照 [该文档](https://aicarrier.feishu.cn/wiki/PzP6wL6d1is9mhkHY3Lc5TYVnJA) 配置 dlc 等工具。根据需要可能要修改 `configs/aliyun_env.py` 中的内容。

其运行命令与 S 集群上相同，唯一差别在于不需要指定 `--slurm` 及相关参数如 `-p` `-q` 等

```bash
python run.py configs/eval.py --max-num-workers 32
```

## 开发规范

见 [飞书文档](https://aicarrier.feishu.cn/wiki/wikcnocfGDlTixegjAgstKP476e)

## FAQ

1. 出现与 NCCL 相关的问题，初步可以尝试在运行的启动命令之前添加: `NCCL_SOCKET_IFNAME=eth0`
