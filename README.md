# Inference-time Scaling of Diffusion Models through Classical Search

## Overview

This repository provides the implementation of [**Inference-time Scaling of Diffusion Models through Classical Search**](https://arxiv.org/abs/2505.23614). The approach leverages classical search algorithms to scale inference compute in diffusion models, improving efficiency and output quality.

## Implementation
The `imagenet` folder provides the implementation of BFS for class-conditional image generation, the `locomotion` folder provides the Q-verifier test-time search for offline RL tasks, and the `pointmaze` folder provides the implementation of the long-horizon planning task. For installation of each task, refer to the instructions in each subfolder.

## Citation

If you use this code, please cite:

```bibtex
@misc{zhang2025inferencetimescalingdiffusionmodels,
      title={Inference-time Scaling of Diffusion Models through Classical Search}, 
      author={Xiangcheng Zhang and Haowei Lin and Haotian Ye and James Zou and Jianzhu Ma and Yitao Liang and Yilun Du},
      year={2025},
      eprint={2505.23614},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23614}, 
}
```

## License

This project is licensed under the MIT License.
