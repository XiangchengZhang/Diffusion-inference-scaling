# Inference-time Scaling of Diffusion Models through Classical Search

## Overview

This repository provide the implementation of [**Inference-time Scaling of Diffusion Models through Classical Search**](https://diffusion-inference-scaling.github.io/). The approach leverages classical search algorithms to scale inference compute in diffusion models, improving efficiency and output quality.

## Implementation
The `imagenet` folder provides the implementation of BFS for class-conditional image generation, the `locomotion` folder provides the Q-verifier test-time search for offline RL tasks, and the `pointmaze` folder provides the implementation of the long-horizon planning task. For installation of each task, refer to the instructions in each subfolder.

## Citation

If you use this code, please cite:

```bibtex
@article{your2024diffusionsearch,
    title={Inference-time Scaling of Diffusion Models through Classical Search},
    author={Your Name},
    journal={arXiv preprint arXiv:xxxx.xxxxx},
    year={2024}
}
```

## License

This project is licensed under the MIT License.
