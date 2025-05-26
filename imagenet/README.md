## ImageNet Class Conditional Generation

The code base is adapted from [Training-Free-Guidance](https://github.com/YWolfeee/Training-Free-Guidance). Please install the environment and download the models following the setup in [setup.md](setup.md). 

To sample BFS, check the scripts in `scripts/search`. You can scale up the compute by changing the number of particles in `per_sample_batch_size` argument. 
