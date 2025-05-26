# TFG: Unified Training-Free Guidance for Diffusion Models

<p align="center">
<!--     </br>
    </br> -->
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.9+-1f425f.svg?color=purple">
    </a> <a href="https://huggingface.co/docs/diffusers">
        <img alt="Diffusers" src="https://img.shields.io/badge/Diffusers-0.26-blue">
    </a> <a>
        <img alt="MIT" src="https://img.shields.io/badge/License-MIT-yellow">
    </a>
</p>

Code and data for our paper [TFG: Unified Training-Free Guidance for Diffusion Models](https://arxiv.org/abs/2409.15761).

## 📰 News
* **[May. 4, 2025]**: We incorporate a Best-of-N feature in `pipeline.py` to better support the inference-time scaling study. You can adjust the `bon_rate` (default=1) to use different "N" in your Best-of-N sampling strategy.
* **[Oct. 30, 2024]**: We launch the first version of **TFG**. The current codebase supports all tasks mentioned in our paper, including label guidance, super resolution, gaussian deblur, fine-grained generation, audio declipping, guidance combinatin, style transfer, and molecule property guidance. More applications & models will be included in the future. 

## 👋 Overview

 <p align="center">    
    <a href="https://github.com/YWolfeee/Training-Free-Guidance">    
        <img src="https://github.com/YWolfeee/Training-Free-Guidance/blob/main/assets/figure.png" width="600"/>    
    </a>       
<p>
    
Given an unconditional diffusion model and a predictor for a target property of interest (e.g., a classifier), the goal of training-free guidance is to generate samples with desirable target properties without additional training. This paper introduces a novel algorithmic framework encompassing existing methods as special cases, unifying the study of training-free guidance into the analysis of an algorithm-agnostic design space. Via theoretical and empirical investigation, we propose an efficient and effective hyper-parameter searching strategy that can be readily applied to any downstream task. We systematically benchmark across 7 diffusion models on 16 tasks with 40 targets, and improve performance by 8.5% on average. Our framework and benchmark offer a solid foundation for conditional generation in a training-free manner.


## 🚀 Set Up
1. **Install packages**. Install packages via `pip install -r requirements.txt`.
2. **Download resources**. Each experiment requires a *diffusion model*, a *guidance function*, and possibly a *dataset* (e.g., for super resolution).
   - You can download all the checkpoints from this [link](https://drive.google.com/drive/folders/1fS7dKpO4O-FjaLwuRXuHBxEOlkqMMTGh?usp=sharing). Set the `MODEL_PATH` in `utils/env_utils.py` as the path of the downloaded directory.

## 💽 Usage
You can check `./scripts` for examplar scripts. Also, if you want to write the script yourself, please refer to `./utils/configs.py` for details.

## 💫 Contributions
We would welcome any contributions, pull requests, or issues!
To do so, please either file a new pull request or issue. We'll be sure to follow up shortly!

## ✍️ Citation
If you find our work helpful, please use the following citations.
```
@article{
    ye2024tfg,
    title={TFG: Unified Training-Free Guidance for Diffusion Models},
    author={Haotian Ye and Haowei Lin and Jiaqi Han and Minkai Xu and Sheng Liu and Yitao Liang and Jianzhu Ma and James Zou and Stefano Ermon},
    booktitle={NeurIPS},
    year={2024}
}
```

## 🪪 License
MIT. Check `LICENSE.md`.
