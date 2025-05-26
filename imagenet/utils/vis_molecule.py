from utils.utils import get_config, get_evaluator
import pickle
import os
from tqdm import tqdm
from utils.vis_utils import plot_ours


if __name__ == '__main__':
    # Please tsee utils/config.py for the complete argument lists
    args = get_config()
    # additional params used in this visualization script:
    # --topk 10
    # --sort_metric mae
    # --saved_file PATH_TO_SAVED_MOLECULE
    # --output_path OUTPUT_PATH_OF_VIS_FIGURES
    # --max_n_samples 1000

    # evaluator for generated samples
    try:
        evaluator = get_evaluator(args)
    except NotImplementedError:
        evaluator = None

    with open(args.saved_file, 'rb') as f:
        samples = pickle.load(f)
    # loop through the batch
    n_samples = min(len(samples), args.max_n_samples)
    samples = samples[:n_samples]
    results = []
    for idx, sample in tqdm(enumerate(samples), total=n_samples):
        metrics = evaluator.evaluate([sample])
        metrics['idx'] = idx
        metrics['target'] = sample[-1][0]
        results.append(metrics)

    # sort
    lower_is_better = args.sort_metric == 'mae'
    results.sort(key=lambda x: x[args.sort_metric], reverse=not lower_is_better)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # plot the best candidates in each bin, hard-coded here for the property alpha
    bins = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]

    for bin_id, bin in enumerate(bins):
        # filter results
        cur_results = []
        for res in results:
            if res['target'] >= bin[0] and res['target'] < bin[1]:
                cur_results.append(res)
        for i in tqdm(range(min(args.topk, len(cur_results)))):
            molecule = samples[cur_results[i]['idx']]
            cur_metric_val = cur_results[i][args.sort_metric]
            cur_target = cur_results[i]['target']
            output_path = os.path.join(args.output_path, f'bin_{bin_id}_rank_{i}_{args.sort_metric}_{cur_metric_val:.3f}'
                                                         f'_target_{cur_target:.3f}.png')
            plot_ours(molecule, output_path=output_path, metric=cur_metric_val)

