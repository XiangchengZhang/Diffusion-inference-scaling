from utils.utils import get_config, get_evaluator, get_guidance, get_network
from pipeline import BasePipeline
import torch
import logger
from copy import deepcopy

if __name__ == '__main__':
    # Please tsee utils/config.py for the complete argument lists
    args = get_config()
    ## prepare core modules based on configs ##
    
    # Unconditional generative model
    network = get_network(args)
    # guidance method encoded by prediction model
    guider = get_guidance(args, network)
    
    bon_guider = None
    if hasattr(args, 'global_verifier') and args.global_verifier:
        global_args = deepcopy(args)
        global_args.guide_networks = args.global_verifier.split('+')
        global_verifier = get_guidance(global_args, network)
    
    # evaluator for generated samples
    try:
        evaluator = get_evaluator(args)
    except NotImplementedError:
        evaluator = None

    pipeline = BasePipeline(args, network, guider, evaluator, global_verifier=global_verifier)

    samples,compute = pipeline.sample(args.num_samples)
    logger.log_samples(samples)
    
    # release torch occupied gpu memory
    torch.cuda.empty_cache()
    
    metrics = evaluator.evaluate(samples)
    for k,v in compute.items():
        metrics[k] = v
    if metrics is not None: # avoid rewriting metrics to json
        logger.log_metrics(metrics, save_json=True)