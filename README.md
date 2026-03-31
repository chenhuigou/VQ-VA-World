<p align="center">
  <a href="https://arxiv.org/abs/2511.20573">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoColor=red" alt="Paper"/>
  </a>
  <a href="https://chenhuigou.github.io/VQ-VA-World">
    <img src="https://img.shields.io/badge/Project-Page-0A66C2?logo=safari&logoColor=white" alt="Project Page"/>
  </a>
  <a href="https://huggingface.co/VQVA">
    <img src="https://img.shields.io/badge/VQVA-Organization-yellow?logo=huggingface&logoColor=yellow" alt="HuggingFace"/>
  </a>
  <a href="https://huggingface.co/datasets/VQVA/IntelligentBench">
    <img src="https://img.shields.io/badge/IntelligentBench-Benchmark-green?logo=huggingface&logoColor=yellow" alt="Benchmark"/>
  </a>
  <a href="https://huggingface.co/datasets/VQVA/BAGEL-World-data">
    <img src="https://img.shields.io/badge/Training-Data-blue?logo=huggingface&logoColor=yellow" alt="Training Data"/>
  </a>
  <a href="https://github.com/chenhuigou/VQ-VA-World">
    <img src="https://img.shields.io/badge/GitHub-Code-black?logo=github&logoColor=white" alt="Code"/>
  </a>
</p>

# VQ-VA World: Towards High-Quality Visual Question-Visual Answering

<p align="center">
<a href="https://www.linkedin.com/in/chenhui-gou-9201081a1/">Chenhui Gou</a><sup>1,4*†</sup>,
<a href="https://scholar.google.com/">Zilong Chen</a><sup>2,4*†</sup>,
<a href="https://zw615.github.io/">Zeyu Wang</a><sup>3*</sup>,
<a href="https://fengli-ust.github.io/">Feng Li</a><sup>4</sup>,
<a href="https://tsutikgiau.github.io/">Deyao Zhu</a><sup>4</sup>,
Zicheng Duan<sup>5</sup>,
<a href="https://andy1621.github.io/">Kunchang Li</a><sup>4</sup>,
<a href="https://scholar.google.com/citations?hl=en&amp;user=k0TWfBoAAAAJ">Chaorui Deng</a><sup>4</sup>,
Hongyi Yuan<sup>4</sup>,
<a href="https://haoqifan.github.io/">Haoqi Fan</a><sup>4</sup>,
<a href="https://cihangxie.github.io/">Cihang Xie</a><sup>3</sup>,
<a href="https://jianfei-cai.github.io/">Jianfei Cai</a><sup>1</sup>,
<a href="https://scholar.google.com.au/citations?user=VxAuxMJAAAAJ">Hamid Rezatofighi</a><sup>1</sup>
</p>

<p align="center">
<sup>1</sup>Monash University, <sup>2</sup>Tsinghua University, <sup>3</sup>UC Santa Cruz, <sup>4</sup>ByteDance Seed, <sup>5</sup>University of Adelaide
</p>

<p align="center"><em>*Equal contribution, †Work done during internship</em></p>

## 📢 News

- **Mar 2026:** Training data released on [HuggingFace](https://huggingface.co/datasets/VQVA/BAGEL-World-data). <a href="https://huggingface.co/datasets/VQVA/BAGEL-World-data"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FVQVA%2FBAGEL-World-data&query=%24.downloads&label=downloads&color=blue&logo=huggingface&logoColor=yellow" alt="downloads"/></a>
- **Mar 2026:** [IntelligentBench](https://huggingface.co/datasets/VQVA/IntelligentBench) benchmark released. <a href="https://huggingface.co/datasets/VQVA/IntelligentBench"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FVQVA%2FIntelligentBench&query=%24.downloads&label=downloads&color=green&logo=huggingface&logoColor=yellow" alt="downloads"/></a>
- **Mar 2026:** Code released.

## 📖 Abstract

We present **VQ-VA World**, a dataset and benchmark for high-quality Visual Question-Visual Answering. Our approach focuses on training models that can understand visual questions and generate visual answers, spanning tasks including intelligent image editing with world knowledge, reasoning, and design knowledge capabilities.

<p align="center">
  <img src="https://chenhuigou.github.io/VQ-VA-World/images/teaser_v2.png" alt="VQ-VA World" width="70%"/>
</p>

## 🔗 Resources

<p align="center">
📄 <a href="https://arxiv.org/abs/2511.20573"><b>Paper</b></a> &nbsp;|&nbsp;
🌐 <a href="https://chenhuigou.github.io/VQ-VA-World"><b>Project Page</b></a> &nbsp;|&nbsp;
🤗 <a href="https://huggingface.co/VQVA"><b>HF Organization</b></a> &nbsp;|&nbsp;
📊 <a href="https://huggingface.co/datasets/VQVA/IntelligentBench"><b>IntelligentBench</b></a> &nbsp;|&nbsp;
📦 <a href="https://huggingface.co/datasets/VQVA/BAGEL-World-data"><b>Training Data</b></a>
</p>

## 🔥 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/chenhuigou/VQ-VA-World.git
cd VQ-VA-World
pip install -r requirements.txt
```

### 2. Download Data

```python
from datasets import load_dataset

# IntelligentBench (benchmark)
bench = load_dataset("VQVA/IntelligentBench", "original", split="test")

# Training data
train = load_dataset("VQVA/BAGEL-World-data", "world_knowledge_uid_filtered")
```

## 📊 IntelligentBench

A benchmark for evaluating intelligent image editing models with 360 samples across three categories.

- **Data**: [VQVA/IntelligentBench](https://huggingface.co/datasets/VQVA/IntelligentBench)
- **Evaluation Script**: [`eval/intelligent/benchmark_intelligentBench.py`](eval/intelligent/benchmark_intelligentBench.py)
- **Launch Script**: [`scripts/eval/launch_intelligent.sh`](scripts/eval/launch_intelligent.sh)

| Category | Code | Samples | Description |
|----------|------|---------|-------------|
| World Knowledge | w | 171 | Editing requiring real-world knowledge |
| Reasoning | r | 101 | Editing requiring logical reasoning |
| Design Knowledge | d | 88 | Editing requiring design expertise |

## 🏗️ Project Structure

```
VQ-VA-World/
├── modeling/           # Model architecture
│   └── lightfusion/    # LightFusion model
├── train/              # Training scripts
├── eval/               # Evaluation scripts
│   ├── i2i/            # Image-to-image evaluation
│   └── t2i/            # Text-to-image evaluation
├── data/               # Data loading & configs
│   └── configs/        # Training configs (stage1, stage2)
└── scripts/            # Launch scripts
```

## ✍️ Citation

If you find VQ-VA World useful for your research, please cite:

```bibtex
@misc{gou2025vqvaworldhighqualityvisual,
      title={VQ-VA World: Towards High-Quality Visual Question-Visual Answering}, 
      author={Chenhui Gou and Zilong Chen and Zeyu Wang and Feng Li and Deyao Zhu and Zicheng Duan and Kunchang Li and Chaorui Deng and Hongyi Yuan and Haoqi Fan and Cihang Xie and Jianfei Cai and Hamid Rezatofighi},
      year={2025},
      eprint={2511.20573},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.20573}, 
}
```

## 📜 License

This project is licensed under [Apache 2.0](LICENSE).

## 🙏 Acknowledgements

This work builds upon [LightFusion](https://arxiv.org/abs/2510.22946). We thank the authors for their excellent work.

Our training data is built from [OmniCorpus-CC](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC). We acknowledge and comply with the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/) and [Terms of Use](https://commoncrawl.org/terms-of-use) of the original dataset.
