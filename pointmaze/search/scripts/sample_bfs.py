from search.methods.bfs import BFSGuidance 
from search.configs import Arguments
from search.search_policy import SearchPolicy
from search.base_pipeline import BasePipe
import csv
import torch
from search.script_utils import get_args

args = Arguments()
args.device = "cuda:7"
args.dataset = 'pointmaze-ultra-navigate-v0'
args.method = 'bfs-resampling'
args_grid = get_args(args)
for args in args_grid:
    args.num_samples = 40
    policy = SearchPolicy(args)
    guidance = BFSGuidance(args=args)
    pipe = BasePipe(args, policy, guidance)
    returns = pipe.experiment()
    success_rate = returns['average']['total_reward']
    total_compute = returns['average']['compute']
    output_file = f'results_{args.method}_{args.dataset}.csv'

    # Write headers if the file does not exist
    try:
        with open(output_file, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'compute', 'recur_steps', 'inference_steps', 'particles', 'rho', 'mu', 'temp', 'start', 'step_size', 
                'success_rate',
            ])
    except FileExistsError:
        pass

    # Append the results
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            total_compute, args.recur_steps, args.inference_steps, args.per_sample_batch_size, args.rho, args.mu, args.temp, args.start_step, args.step_size,
            success_rate,
        ])