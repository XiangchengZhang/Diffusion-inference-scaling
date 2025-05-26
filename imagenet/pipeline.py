import torch
import os
import numpy as np
import PIL.Image as Image
from abc import ABC, abstractmethod
from diffusion.base import BaseSampler
from methods.base import BaseGuidance
from evaluations.base import BaseEvaluator
from utils.configs import Arguments
import logger

class BasePipeline(object):
    def __init__(self,
                 args: Arguments, 
                 network: BaseSampler, 
                 guider: BaseGuidance, 
                 evaluator: BaseEvaluator,
                 global_verifier=None):
        self.network = network
        self.guider = guider
        self.evaluator = evaluator
        self.logging_dir = args.logging_dir
        self.check_done = args.check_done
        
        self.batch_size = args.eval_batch_size
        
        # init global verifier for double verifier
        self.global_verifier = global_verifier if global_verifier is not None else self.guider
        
    @abstractmethod
    def sample(self, sample_size: int):
        
        load_samples = self.check_done_and_load_sample()
        if load_samples is not None:
            logger.log("Loaded samples from previous run.")
            samples, compute = load_samples
        else:
            samples, compute = None, {'compute': 0}
        
        if samples is None:
            samples, compute = self.network.sample(sample_size=sample_size, guidance=self.guider, global_verifier=self.global_verifier)
            samples = self.network.tensor_to_obj(samples)
                    
        return samples, compute
    
    def evaluate(self, samples):
        return self.check_done_and_evaluate(samples)
    
    def check_done_and_evaluate(self, samples):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, 'metrics.json')):
            logger.log("Metrics already generated. To regenerate, please set `check_done` to `False`.")
            return None
        return self.evaluator.evaluate(samples)

    def check_done_and_load_sample(self):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, "finished_sampling")):
            logger.log("found tags for generated samples, should load directly. To regenerate, please set `check_done` to `False`.")
            return logger.load_samples()

        return None

