<div align="center">

# 3D-GAD

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This repo is the official implementation of our work "3D Generative Model Reveals the Heterogeneity of Neuroanatomical Subtypes within Alzheimer’s Disease"

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10.0
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```
## Data 

The data used in this study includes three public datasets. To access these datasets, you will need to register and submit a request on their respective websites.
1. ADNI：www.adni-info.org
2. AIBL: [aibl.csiro.au/adni/index.html](https://aibl.csiro.au/adni/index.html)
3. OASIS-3: www.oasis-brains.org/#access)
After obtaining the data, you will need to preprocess the datasets according to the methods described in our paper.

## How to run


Train regular diffusion

```bash
sh ./scripts/train_contrastive_diffusion_regular.sh
```

Train contrastive diffusion

```bash
sh ./scripts/train_contrastive_diffusion.sh
```
