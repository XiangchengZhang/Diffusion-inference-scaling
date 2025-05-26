'''
Experiment script for inference-time search
'''
from pipeline import Pipeline
from search.configs import Arguments
from search.utils import get_hyper_params
import csv
import os





datasets = ["halfcheetah-medium-expert-v2", "walker2d-medium-expert-v2", "hopper-medium-expert-v2",
            "halfcheetah-medium-replay-v2", "walker2d-medium-replay-v2", "hopper-medium-replay-v2",
            "halfcheetah-medium-v2", "walker2d-medium-v2", "hopper-medium-v2",]

def experiment(dataset="halfcheetah-medium-expert-v2", device="cuda:0", seed=0):
    args = Arguments()
    args.dataset = dataset
    args.device = device
    args.seed = seed
    args_grid = get_hyper_params(args)

    ## log the results with csv
    output_file = f"results/{args.dataset}/inference_seed.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["seed", "particles", "recur_steps","iter_steps", "rho", "mu","sigma", "mean", "std"])
    
    for args in args_grid:
        pipeline = Pipeline(args)
        mean, std = pipeline.eval()
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([seed, args.per_sample_batch_size, args.recur_steps, args.iter_steps, args.rho, args.mu, args.sigma, mean, std])
    
    
    

if __name__ == "__main__":
    # datasets = ["walker2d-medium-replay-v2",]
    import multiprocessing
    processes = []
    for ds in datasets:
        for seed in range(5):
            p = multiprocessing.Process(target=experiment, kwargs={"dataset": ds, "device": "cuda:5", "seed": seed})
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    # experiment()