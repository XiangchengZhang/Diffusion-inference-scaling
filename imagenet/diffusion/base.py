from abc import ABC, abstractmethod
import torch
from typing import List
from methods.base import BaseGuidance

class BaseSampler:

    def __init__(self, args):
        self.device = args.device
        self.seed = args.seed
        self.per_sample_batch_size = args.per_sample_batch_size
        
    @abstractmethod
    def sample(self, sample_size: int, guidance: BaseGuidance):
        pass

    @staticmethod
    def tensor_to_obj(tensor: torch.tensor):
        # This function taks into the generated torch tensor and converts it into a list of objects in the corresponding domain
        pass

    @staticmethod
    def obj_to_tensor(objs: List):
        # This function takes in a list of objects and converts it into a torch tensor
        pass