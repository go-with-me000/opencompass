<div align="center">
  <img src="docs/en/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![docs](https://readthedocs.org/projects/opencompass/badge)](https://opencompass.readthedocs.io/en)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](https://github.com/InternLM/opencompass/blob/main/LICENSE)

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐Website](https://opencompass.org.cn/) |
[📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started.html#installation) |
[🤔Reporting Issues](https://github.com/InternLM/opencompass/issues/new/choose)

English | [简体中文](README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/KKwfEbFj7U" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=opencompass" target="_blank">WeChat</a>
</p>

## 🧭	Welcome

to **OpenCompass**!

Just like a compass guides us on our journey, OpenCompass will guide you through the complex landscape of evaluating large language models. With its powerful algorithms and intuitive interface, OpenCompass makes it easy to assess the quality and effectiveness of your NLP models.

> **🔥 Attention**<br />
> We launch the OpenCompass Collabration project, welcome to support diverse evaluation benchmarks into OpenCompass!
> Clike [Issue](https://github.com/InternLM/opencompass/issues/248) for more information.
> Let's work together to build a more powerful OpenCompass toolkit!

## 🚀 What's New <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2023.08.25\]**  [**TigerBot**](https://github.com/TigerResearch/TigerBot) team adpots OpenCompass to evaluate their models systematically. We deeply appreciate the community's dedication to transparency and reproducibility in LLM evaluation.
- **\[2023.08.21\]** [**Lagent**](https://github.com/InternLM/lagent) has been released, which is a lightweight framework for building LLM-based agents. We are working with Lagent team to support the evaluation of general tool-use capability, stay tuned! 🔥🔥🔥.
- **\[2023.08.18\]** We have supported evaluation for **multi-modality learning**, include **MMBench, SEED-Bench, COCO-Caption, Flickr-30K, OCR-VQA, ScienceQA** and so on. Leaderboard is on the road. Feel free to try multi-modality evaluation with  OpenCompass ! 🔥🔥🔥.
- **\[2023.08.18\]** [Dataset card](https://opencompass.org.cn/dataset-detail/MMLU) is now online. Welcome new evaluation benchmark  OpenCompass !
- **\[2023.08.11\]** [Model comparison](https://opencompass.org.cn/model-compare/GPT-4,ChatGPT,LLaMA-2-70B,LLaMA-65B) is now online. We hope this feature offers deeper insights!
- **\[2023.08.11\]** We have supported [LEval](https://github.com/OpenLMLab/LEval).

> [More](docs/en/notes/news.md)

## ✨ Introduction

OpenCompass is a one-stop platform for large model evaluation, aiming to provide a fair, open, and reproducible benchmark for large model evaluation. Its main features includes:

- **Comprehensive support for models and datasets**: Pre-support for 20+ HuggingFace and API models, a model evaluation scheme of 50+ datasets with about 300,000 questions, comprehensively evaluating the capabilities of the models in five dimensions.

- **Efficient distributed evaluation**: One line command to implement task division and distributed evaluation, completing the full evaluation of billion-scale models in just a few hours.

- **Diversified evaluation paradigms**: Support for zero-shot, few-shot, and chain-of-thought evaluations, combined with standard or dialogue type prompt templates, to easily stimulate the maximum performance of various models.

- **Modular design with high extensibility**: Want to add new models or datasets, customize an advanced task division strategy, or even support a new cluster management system? Everything about OpenCompass can be easily expanded!

- **Experiment management and reporting mechanism**: Use config files to fully record each experiment, support real-time reporting of results.

## 📊 Leaderboard

We provide [OpenCompass Leaderbaord](https://opencompass.org.cn/rank) for community to rank all public models and API models. If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address `opencompass@pjlab.org.cn`.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📖 Dataset Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Language</b>
      </td>
      <td>
        <b>Knowledge</b>
      </td>
      <td>
        <b>Reasoning</b>
      </td>
      <td>
        <b>Comprehensive Examination</b>
      </td>
      <td>
        <b>Understanding</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Word Definition</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>Idiom Learning</b></summary>

- CHID

</details>

<details open>
<summary><b>Semantic Similarity</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>Coreference Resolution</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>Translation</b></summary>

- Flores

</details>
      </td>
      <td>
<details open>
<summary><b>Knowledge Question Answering</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestion
- TrivialQA

</details>

<details open>
<summary><b>Multi-language Question Answering</b></summary>

- TyDi-QA

</details>
      </td>
      <td>
<details open>
<summary><b>Textual Entailment</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE

</details>

<details open>
<summary><b>Commonsense Reasoning</b></summary>

- StoryCloze
- StoryCloze-CN (coming soon)
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>Mathematical Reasoning</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>Theorem Application</b></summary>

- TheoremQA

</details>

<details open>
<summary><b>Code</b></summary>

- HumanEval
- MBPP

</details>

<details open>
<summary><b>Comprehensive Reasoning</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>Junior High, High School, University, Professional Examinations</b></summary>

- GAOKAO-2023
- CEval
- AGIEval
- MMLU
- GAOKAO-Bench
- CMMLU
- ARC

</details>
      </td>
      <td>
<details open>
<summary><b>Reading Comprehension</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE

</details>

<details open>
<summary><b>Content Summary</b></summary>

- CSL
- LCSTS
- XSum

</details>

<details open>
<summary><b>Content Analysis</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📖 Model Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Open-source Models</b>
      </td>
      <td>
        <b>API Models</b>
      </td>
      <!-- <td>
        <b>Custom Models</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>

- InternLM
- LLaMA
- Vicuna
- Alpaca
- Baichuan
- WizardLM
- ChatGLM-6B
- ChatGLM2-6B
- MPT
- Falcon
- TigerBot
- MOSS
- ...

</td>
<td>

- OpenAI
- Claude (coming soon)
- PaLM (coming soon)
- ……

</td>

<!--
- GLM
- ...

</td> -->

</tr>
  </tbody>
</table>

## 🛠️ Installation

Below are the steps for quick installation and datasets preparation.

```Python
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/InternLM/opencompass opencompass
cd opencompass
pip install -e .
# Download dataset to data/ folder
wget https://github.com/InternLM/opencompass/releases/download/0.1.1/OpenCompassData.zip
unzip OpenCompassData.zip
```

Some third-party features, like Humaneval and Llama, may require additional steps to work properly, for detailed steps please refer to the [Installation Guide](https://opencompass.readthedocs.io/en/latest/get_started.html).

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🏗️ ️Evaluation

After ensuring that OpenCompass is installed correctly according to the above steps and the datasets are prepared, you can evaluate the performance of the LLaMA-7b model on the MMLU and C-Eval datasets using the following command:

```bash
python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl
```

OpenCompass has predefined configurations for many models and datasets. You can list all available model and dataset configurations using the [tools](./docs/en/tools.md#list-configs).

```bash
# List all configurations
python tools/list_configs.py
# List all configurations related to llama and mmlu
python tools/list_configs.py llama mmlu
```

You can also evaluate other HuggingFace models via command line. Taking LLaMA-7b as an example:

```bash
python run.py --datasets ceval_ppl mmlu_ppl \
--hf-path huggyllama/llama-7b \  # HuggingFace model path
--model-kwargs device_map='auto' \  # Arguments for model construction
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # Arguments for tokenizer construction
--max-out-len 100 \  # Maximum number of tokens generated
--max-seq-len 2048 \  # Maximum sequence length the model can accept
--batch-size 8 \  # Batch size
--no-batch-padding \  # Don't enable batch padding, infer through for loop to avoid performance loss
--num-gpus 1  # Number of required GPUs
```

Through the command line or configuration files, OpenCompass also supports evaluating APIs or custom models, as well as more diversified evaluation strategies. Please read the [Quick Start](https://opencompass.readthedocs.io/en/latest/get_started.html) to learn how to run an evaluation task.

## 🔜 Roadmap

- [ ] Subjective Evaluation
  - [ ] Release CompassAreana
  - [ ] Subjective evaluation dataset.
- [ ] Long-context
  - [ ] Long-context evaluation with extensive datasets.
  - [ ] Long-context leaderboard.
- [ ] Coding
  - [ ] Coding evaluation leaderdboard.
  - [ ] Non-python language evaluation service.
- [ ] Agent
  - [ ] Support various agenet framework.
  - [ ] Evaluation of tool use of the LLMs.
- [ ] Robustness
  - [ ] Support various attack method

## 👷‍♂️ Contributing

We appreciate all contributions to improve OpenCompass. Please refer to the [contributing guideline](https://opencompass.readthedocs.io/en/latest/notes/contribution_guide.html) for the best practice.

## 🤝 Acknowledgements

Some code in this project is cited and modified from [OpenICL](https://github.com/Shark-NLP/OpenICL).

Some datasets and prompt implementations are modified from [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) and [instruct-eval](https://github.com/declare-lab/instruct-eval).

## 🖊️ Citation

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/InternLM/OpenCompass}},
    year={2023}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>
