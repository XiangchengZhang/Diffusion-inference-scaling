## PointMaze Planning

The code is based on [diffuser](https://github.com/jannerm/diffuser) and the maze environment is based on [ogbench](https://github.com/seohongpark/ogbench). 
### Install
First install the dependencies in `requirements.txt`, then install ogbench with our added PointMaze Ultra environment. 
```bash
conda create -n maze python=3.10
pip install -r requirements.txt
cd ogbench
pip install -e .
cd ..
pip install -e .
```
Then, download and extract the pretrained models from [here](https://drive.google.com/file/d/1ZMhoOkLLMozUdADKph3oed_OwzAYtiD1/view?usp=sharing) and put them in the `logs/` directory. For the collected trajectory dataset for the Ultra Maze, download and extract the files from [here](https://drive.google.com/file/d/1doAvARCm04axeXFUn8ETRgWfcoMt84m3/view?usp=sharing), and put the files `pointmaze-ultra-navigate-v0.npz` and `pointmaze-ultra-navigate-v0-val.npz` in the directory `~/.ogbench/data`. 

### Inference
For inference scaling, run 
```bash
python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --device cuda
```
You can change the method in `dfs`, `bfs-resampling`, `bfs-pruning` and `bon`, and change the dataset in `pointmaze-giant-navigate-v0` and `pointmaze-ultra-navigate-v0`. 

