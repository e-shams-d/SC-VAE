[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14165631-blue.svg)](https://doi.org/10.1016/j.patcog.2024.111187)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)

# [SC-VAE: Sparse coding-based variational autoencoder with learned ISTA](https://www.sciencedirect.com/science/article/pii/S0031320324009385)


## Introduction
This repository contains official implementation for the paper titled "SC-VAE: Sparse Coding-based Variational Autoencoder with Learned ISTA".

## Installing Dependencies
To install dependencies, create a conda or virtual environment with Python 3 and then run `pip install -r requirements.txt`.

## Training the SC-VAE
To run the SC-VAE simply run `python main-stage1.py`. You could change the config files in `line 279` to train SC-VAE model with different downsampling blocks.
```python
parser.add_argument('--model-config', type=str, default='./configs/ffhq/stage1/ffhq256-scvae16x16.yaml')
```

## Citation
@article{xiao2023sc,    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={SC-VAE: Sparse Coding-based Variational Autoencoder with Learned ISTA},    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Xiao, Pan and Qiu, Peijie and Ha, Sung Min and Bani, Abdalla and Zhou, Shuang and Sotiras, Aristeidis},    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal={Pattern Recognition},    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2025}    
}

## To-Do List
- [x] Installing dependencies
- [x] Training the Model
- [] Uploading pre-trained code