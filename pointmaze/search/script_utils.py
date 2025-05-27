from search.configs import Arguments
from search.search_policy import SearchPolicy
from search.methods.bfs import BFSGuidance
from search.methods.dfs import DFSGuidance
from search.base_pipeline import BasePipe
from copy import deepcopy
from typing import List


def get_pipe(args: Arguments) -> BasePipe:
    if args.method == 'dfs':
        guidance = DFSGuidance(args=args)
    elif 'bfs' in args.method or 'bon' in args.method:
        guidance = BFSGuidance(args=args)
    else:
        raise NotImplementedError(f"Method {args.method} is not implemented.")
    policy = SearchPolicy(args)
    pipe = BasePipe(args, policy, guidance)
    return pipe


def get_args(args: Arguments) -> List[Arguments]:
    args_grid = []
    if args.dataset == 'pointmaze-giant-navigate-v0':
        args.rho = 0.04
        args.mu = 0.01
        args.sigma = 0.00
        args.recur_steps = 2
        args.inference_steps = 16
        args.eps_bsz = 1
        args.sampling_horizon = 600
        if args.method == 'bon':
            args.temp = 0.0
            for particles in [4, 8, 16, ]:
                cur_args = deepcopy(args)
                cur_args.per_sample_batch_size = particles
                args_grid.append(cur_args)
            return args_grid
        elif 'bfs' in args.method:
            args.temp = 0.02
            args.temp_schedule = 'increase'
            args.start_step = 4
            args.step_size = 4
            for particles in [4, 8, 16, ]:
                cur_args = deepcopy(args)
                cur_args.per_sample_batch_size = particles
                args_grid.append(cur_args)
            return args_grid
        elif 'dfs' in args.method:
            args.per_sample_batch_size = 1
            args.step_size = 1
            args.threshold_schedule = 'increase'
            for (depth, budget, threshold) in [(12, 20, 6), (12, 20, 4), (14, 20, 2)]:
                cur_args = deepcopy(args)
                cur_args.budget = budget
                cur_args.threshold = threshold
                cur_args.recur_depth = depth
                cur_args.start_step = depth
                args_grid.append(cur_args)
            return args_grid
    elif args.dataset == 'pointmaze-ultra-navigate-v0':
        args.rho = 0.002
        args.mu = 0.002
        args.sigma = 0.01
        args.eps_bsz = 2
        args.recur_steps = 6
        args.inference_steps = 16
        args.sampling_horizon = 2800
        if args.method == 'bon':
            args.temp = 0.0
            for particles in [8, 16, 32]:
                cur_args = deepcopy(args)
                cur_args.per_sample_batch_size = particles
                args_grid.append(cur_args)
            return args_grid
        if 'bfs' in args.method:
            args.temp = 0.005
            args.temp_schedule = 'increase'
            args.start_step = 4
            args.step_size = 4
            for particles in [16, 32]:
                cur_args = deepcopy(args)
                cur_args.per_sample_batch_size = particles
                args_grid.append(cur_args)
            return args_grid
        elif 'dfs' in args.method:
            args.per_sample_batch_size = 1
            args.step_size = 1
            args.start_step = 5
            args.threshold_schedule = 'increase'
            args.threshold = 400
            args.recur_depth = 5
            for budget in [16, 32]:
                cur_args = deepcopy(args)
                cur_args.budget = budget
                args_grid.append(cur_args)
            return args_grid
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

