from search.configs import Arguments
from search.script_utils import get_pipe, get_args

def main(dataset: str="pointmaze-giant-navigate-v0", method: str='dfs', device: str="cuda:7"):
    args = Arguments()
    args.device = device
    args.dataset = dataset
    args.method = method
    args_grid = get_args(args)
    
    for args in args_grid:
        pipe = get_pipe(args)
        returns = pipe.experiment()
        success_rate = returns['average']['total_reward']
        average_compute = returns['average']['compute']
        print(f"Success Rate: {success_rate}, Average Compute: {average_compute}")
        output_file = f'results_{args.method}_{args.dataset}.txt'
        with open(output_file, 'a') as f:
            f.write(f"Maze: {args.dataset} | Method: {args.method} | Compute: {average_compute} | Success Rate: {success_rate}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference scaling experiment.")
    parser.add_argument('--dataset', 
                        type=str, 
                        default='pointmaze-ultra-navigate-v0', 
                        choices=['pointmaze-giant-navigate-v0', 'pointmaze-ultra-navigate-v0'],
                        help='Maze env to use for the experiment.')
    parser.add_argument('--method', 
                        type=str, 
                        default='bon', 
                        choices=['dfs', 'bon', 'bfs-resampling', 'bfs-pruning'],
                        help='Search method to use')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        )
    cli_args = parser.parse_args()

    main(dataset=cli_args.dataset, method=cli_args.method, device=cli_args.device)