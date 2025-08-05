<h1> PreTPP </h1>

<div align="center">

  <a href="">[![Build Status](https://github.com/ivan-chai/pretpp/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ivan-chai/pretpp/actions)</a>
  <a href="">[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)</a>

</div>

<h4 align="center">
    <p>
        <a href="#Installation">Installation</a> |
        <a href="#Data-preparation">Data preparation</a> |
        <a href="#Usage">Usage</a> |
        <a href="#HT-Transformer">HT-Transformer</a> |
        <a href="https://arxiv.org/pdf/2508.01474v1">HT-Transformer (paper)</a> |
        <a href="#Citation">Citing</a>
    <p>
</h4>
Advanced pretraining for TPP, MTPP and Event Sequences.

The code is highly dependent on and compatible with [HoTPP](https://github.com/ivan-chai/hotpp-benchmark).

# Installation

```bash
pip install --no-build-isolation .
```

# HT-Transformer
The code for HT-Transformer can be found at:
```
pretpp/nn/encoder/history_token_transformer.py
pretpp/nn/encoder/history_token_strategy.py
```

# Data preparation
Some datasets are inherited from [HoTPP](https://github.com/ivan-chai/hotpp-benchmark). For them just make a symlink to the data folder:
```bash
cd experiments/DATASET
ln -s <hotpp>/experiments/DATASET/data .
```

To make datasets, specific to PreTPP, use the following command:
```bash
cd experiments/DATASET
spark-submit --driver-memory 16g -c spark.network.timeout=100000s --master 'local[12]' scripts/make-dataset.py
```

# Parameters
All configs are placed at experiments/DATASET/configs.

# Results
All results are stored in experiments/DATASET/results.

# Usage
Example training of HT-Transformer on the Churn dataset:
```bash
cd experiments/transactions-rosbank-full-3s
CUDA_VISIBLE_DEVICES=0 python3 -m hotpp.train_multiseed --config-dir configs --config-name next_item_hts_transformer
```

Fine-tune:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hotpp.train_multiseed --config-dir configs --config-name htl_transformer_ft_multi base_name=next_item_hts_transformer
```

Example training of NTP-Transformer on the Taobao dataset:

```bash
cd experiments/taobao
CUDA_VISIBLE_DEVICES=0 python3 -m hotpp.train_multiseed --config-dir configs --config-name next_item_transformer
```

Fine-tune:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hotpp.train_multiseed --config-dir configs --config-name transformer_ft_multi base_name=next_item_transformer
```

# Citation
```
@article{karpukhin2025httransformer,
  title={HT-Transformer: Event Sequences Classification by Accumulating Prefix Information with History Tokens},
  author={Karpukhin, Ivan and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2508.01474v1},
  year={2025},
  url ={https://arxiv.org/abs/2508.01474v1}
}
```
