import sys
import json
import argparse
import os
from typing import List, Tuple, Any
from utils.configs import Arguments
import copy
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import numpy as np
from functools import cmp_to_key, partial
from utils.utils import get_config, get_logging_dir
DEBUG = False
ALLOW_ERROR_RUN = True

def read_metrics(logging_dir):
        print(f"Trying reading metrics from {logging_dir}")
        metric_json = os.path.join(logging_dir, 'metrics.json')
        with open(metric_json, 'r') as f:
            metrics = json.load(f)
        return metrics

def run_exp(args):
    args, cuda_ids = args
    # cuda_ids is a queue that contains the cuda ids that are available for running the experiments

    logging_dir = get_logging_dir(args)

    # if the logging dir already exists, we read the metrics and return
    try:
        metrics = read_metrics(logging_dir)
        print(f"In side `run_exp`: Metrics already exists in {logging_dir}, return")
        return metrics
    except:
        os.makedirs(logging_dir, exist_ok=True)

    cuda_ids: Queue = cuda_ids
    cuda_id = cuda_ids.get()
    print(multiprocessing.current_process(), cuda_id)

    # prepare running command
    running_command = f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py" 
    for k, v in args.items():
        running_command += f" --{k} '{v}'"
    running_command += f" > {logging_dir}/stdout.log 2>&1"
    
    # run the command
    print('In side `run_exp`:', running_command)
    if not DEBUG:
        os.system(running_command)
    print(f"In side `run_exp`: Finished running and saved to {logging_dir}. Ready to put cuda_id back to queue")
    cuda_ids.put(cuda_id, block=False)
    print("In side `run_exp`: finish put, return metrics...")

    if not DEBUG:    
        try:
            metrics = read_metrics(logging_dir)
            print(f"Metrics: {metrics}")
        except FileNotFoundError:
            if ALLOW_ERROR_RUN:
                print(f"Error running {running_command}, return worst config")
                metrics = {'mae': 10000000, 'molecule_stability': 0, 'atom_stability': 0, 'validity': 0, 'uniqueness': 0.0, 'novelty': 0.0}
            else:
                raise FileNotFoundError(f"Error running {running_command}")
    else:
        metrics = {'validity': cuda_id + args['rho'] + args['mu'] + args['sigma']} if args['guidance_name'] == 'tfg' else {'validity': cuda_id + args['guidance_strength']}

    return metrics

def metrics_key(metrics, keys: List[str]):
    if metrics is None:
        return -100000  # A very large negative number
    a = []
    for key in keys:
        if key in metrics:
            a.append(metrics[key])
        elif key[1:] in metrics and key[0] == '_':
            a.append(-metrics[key[1:]]) # smaller results are better

    return np.mean(a) 
    # return metrics['validity']

def metrics_better(metrics1, metrics2):
    # Placeholder for function that compares two metrics and returns True if metrics1 is better
    return metrics1['validity'] > metrics2['validity']

class BeamSearch:
    def __init__(self, base_args, sweep_params, search_config):
        self.base_args = base_args
        self.config = search_config
        self.sweep_params = sweep_params
        
    def init_args(self, args):
        new_args = copy.deepcopy(args)
        for param, value in self.sweep_params.items():
            new_args[param] = value['init']

        return new_args

    def generate_candidates(self, 
                            args_list: List[Arguments],
                            exist_args_list: List[Arguments] = None):
        exist_args_list = [] if exist_args_list is None else exist_args_list
        candidates = []
        for args in args_list:
            for param, vals in self.sweep_params.items():
                _, factor, max_val = vals
                value = min(args[param] * vals['factor'], vals['max'])
                new_args = copy.deepcopy(args)
                new_args[param] = value
                # If there exists a candidate with the same parameters of rho, mu, sigma, we don't add it to the candidates
                if new_args not in exist_args_list and new_args not in candidates:
                    candidates.append(new_args)
                
                
        return candidates

    def run_candidates(self, args_list: List[Arguments]) -> List[Tuple[Arguments, Any]]:
        results = []    # we use list to maintain the order of the results
        # run the experiments in a multi-processing ways, where cuda_id is the index that can be used to select the cuda device for parallel computation.

        manager = Manager()
        cuda_ids = manager.Queue()
        for cuda_id in self.config.cuda_ids:
            cuda_ids.put(cuda_id)

        p = multiprocessing.Pool(len(self.config.cuda_ids)) 
        results = p.map(run_exp, [(args, cuda_ids) for args in args_list])
        p.close()
        p.join()
        
        assert len(args_list) == len(results) 
        return [(args, metrics) for args, metrics in zip(args_list, results)]
    
    def sorted_results(self, results: List[Tuple[Arguments, Any]]) -> List[Tuple[Arguments, Any]]:
        # Convert the comparator to a key function
        # key_func = cmp_to_key(lambda x, y: metrics_better(x[1], y[1]))
        
        # Sort using the key function derived from the comparator
        sorted_items = sorted(results, key=lambda x: metrics_key(x[1], keys=self.config.metrics), reverse=True)
        
        return sorted_items

    def run(self):
        exist_results = []  # list of ordered tuples of (args, metrics)
        init_candidate = [self.init_args(self.base_args)]
        new_candidates = init_candidate + self.generate_candidates(init_candidate, init_candidate)

        for _ in range(self.config.max_sweep):
            print(f'Beam run {_}. Running new candidates:')
            # print new candidates to be run
            for args in new_candidates:
                print({k: v for k, v in args.items() if k in self.sweep_params.keys()})

            new_results = self.run_candidates(new_candidates)
            exist_results = self.sorted_results(exist_results + new_results)

            print(f'Current number of run results is {len(exist_results)}. Below are the top {self.config.topk} results:')
            for args, metrics in exist_results[: self.config.topk]:
                print({k: v for k, v in args.items() if k in self.sweep_params.keys()}, metrics)
            # print(f"Current top {self.config.topk} results: {exist_results}")


            new_candidates = self.generate_candidates([args for args, _ in exist_results[: self.config.topk]], [args for args, _ in exist_results])

        return exist_results


if __name__ == '__main__':
    argv = sys.argv
    # exp is for main.py, search is for searching.py
    exp_argv, search_argv = argv[:1], argv[:1]
    # for each two arguments in argv, we check whether the key is an attribute of Arguments, and put the pair in exp_argv or search_argv
    args_starts = [i for i, arg in enumerate(argv) if arg.startswith('--')]
    for idx, pos in enumerate(args_starts):
        nxt_pos = args_starts[idx+1] if idx+1 < len(args_starts) else len(argv)
        if hasattr(Arguments, argv[pos][2:]):  # the first two tokens are `--`
            exp_argv.extend(argv[pos:nxt_pos])
        else:
            search_argv.extend(argv[pos:nxt_pos])
    print(exp_argv, search_argv)
    
    sys.argv = exp_argv
    args = get_config(add_logger=False)

    # convert args to dict
    args = {k: v for k, v in vars(args).items() if type(v) in [str, int, float, bool]}
    # exit()

    sys.argv = search_argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', type=str, default='sweeps')
    parser.add_argument('--max_sweep', type=int, default=6)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--cuda_ids', nargs="+", type=int, default=[0])
    parser.add_argument('--beam_sample_size', type=float, default=64, 
                        help='The number of samples to be generated in each iteration when we are doing beam search.')
    parser.add_argument('--num_large_scale', type=int, default=5, help='Run large scale experiments for how many configs after beam search.')
    parser.add_argument('--metrics', type=str, nargs="+", default=['validity'])
    
    if args['guidance_name'] == 'tfg':
        search_params = ['rho', 'mu', 'sigma']
    else:
        search_params = ['guidance_strength']
    
    # parameters for sweeping
    parser.add_argument('--init_rho', type=float, default=0.25)
    parser.add_argument('--factor_rho', type=float, default=2)
    parser.add_argument('--max_rho', type=float, default=8)
    parser.add_argument('--init_mu', type=float, default=0.25)
    parser.add_argument('--factor_mu', type=float, default=2)
    parser.add_argument('--max_mu', type=float, default=8)
    parser.add_argument('--init_sigma', type=float, default=0.001)
    parser.add_argument('--factor_sigma', type=float, default=10)
    parser.add_argument('--max_sigma', type=float, default=1)
    parser.add_argument('--init_guidance_strength', type=float, default=0.25)
    parser.add_argument('--factor_guidance_strength', type=float, default=2)
    parser.add_argument('--max_guidance_strength', type=float, default=8)
    sweep_args = parser.parse_args()
    
    # `args` is the default arguments that are passed to each single run
    # `seeep_args` contains parameters specified for the sweep, and some parameters are pluged into `args` upon the corresponding run
    # `sweep_args` mainly contain two types of configs for each hyper-parameters
    # to be sweeped: init_x and `factor_x`.
    # `init_x` is the initial value of the hyper-parameter to be sweeped.
    # `factor_x` is the multiplier to update the hyper-parameter.
    # When `factor_x` == 1, it's automatically removed from the searching space.
    # `max_x` is the maximum value of the hyper-parameter to be sweeped.
     
    # compute the dict of parameters to be sweeped
    sweep_params = {
        w: {k: getattr(sweep_args, f'{k}_{w}') for k in ['init', 'factor', 'max']} for w in search_params 
    }

    print("Sweeping parameters: ", sweep_params)

    # save for large scale use
    real_num_samples = int(args['num_samples'])        
    real_logging_dir = args['logging_dir']
    print("Save large scale num_samples, logging dir")
    print(f"real_num_samples: {real_num_samples}, real_logging_dir: {real_logging_dir}")

    # add globally invariant parameters into args
    args['num_samples'] = int(sweep_args.beam_sample_size)
    args['logging_dir'] = sweep_args.sweep_dir
    print("Default arguments", args)

    logging_dir = os.path.join(*get_logging_dir(args).split("/")[:-1])
    Search = BeamSearch(args, sweep_params, sweep_args)
    sorted_settings = Search.run()

    print(f"Finished beam search for {sweep_args.max_sweep} run and get {len(sorted_settings)} in totals. saved to {logging_dir}")
    for i, (one_args, metrics) in enumerate(sorted_settings):
        dic = {k: v for k, v in one_args.items() if k in search_params}
        print(f"Top {i+1}: {dic}, {metrics}")

    with open(os.path.join(logging_dir, 'sorted_settings.json'), 'w') as f:
        json.dump(sorted_settings, f, indent=4)

    if sweep_args.num_large_scale == 0:
        print("args.num_large_scale is 0, exit without running large scale experiments.")
        exit()

    # Run large scale experiments for the top k settings
    args_list = [one_args for one_args, _ in sorted_settings[:sweep_args.num_large_scale]]
    for args in args_list:
        args['num_samples'] = real_num_samples
        args['logging_dir'] = real_logging_dir

    print(f"Running large scale experiments for the top {sweep_args.num_large_scale} settings")
    results = Search.run_candidates(args_list)
    print(f"Finished running large scale experiments for the top {sweep_args.num_large_scale} settings")
    
    with open(os.path.join(logging_dir, 'large_scale_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
