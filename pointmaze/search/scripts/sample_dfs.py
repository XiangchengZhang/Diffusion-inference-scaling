from search.methods.dfs import DFSGuidance
from search.search_policy import SearchPolicy
from search.configs import Arguments
from search.base_pipeline import BasePipe
import csv
from search.script_utils import get_args

args = Arguments()
args.device = "cuda:7"
args.dataset = 'pointmaze-giant-navigate-v0'
args.method = 'dfs'
args_grid = get_args(args)
for args in args_grid:
    args.num_samples = 40
    policy = SearchPolicy(args)
    guidance = DFSGuidance(args=args)
    pipe = BasePipe(args, policy, guidance)
    returns = pipe.experiment()
    success_rate = returns['average']['total_reward']
    average_compute = returns['average']['compute']
    output_file = f'results_{args.method}_{args.dataset}.csv'

    # Write headers if the file does not exist
    try:
        with open(output_file, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ 'compute','inference_steps','recur_steps', 'start', 'step_size','budget',  'recur_depth','threshold', 'success_rate'])
    except FileExistsError:
        pass

    # Append the results
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ average_compute, args.inference_steps, args.recur_steps, args.start_step, args.step_size, args.budget, args.recur_depth, args.threshold, success_rate])
