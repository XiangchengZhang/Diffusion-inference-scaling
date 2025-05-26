from search.configs import Arguments
import numpy as np
import torch
from collections import namedtuple
from diffuser.utils import to_np
from diffuser.guides.policies import Policy
from diffuser.models.diffusion import GaussianDiffusion

Trajectories = namedtuple('Trajectories', 'actions observations')

class BasePolicy:
    def __init__(self, args:Arguments, **kwargs):
        self.args = args
        self.device = torch.device(self.args.device)
        self.generator = torch.manual_seed(self.args.seed)
        self.is_setup = False
    
    def setup(self, policy: Policy, diffusion: GaussianDiffusion, **kwargs):
        self.policy = policy
        self.diffusion = diffusion
        self.diffusion.to(self.device)
        
        observations_maxs = self.policy.normalizer.normalizers["observations"].maxs
        observations_mins = self.policy.normalizer.normalizers["observations"].mins
        actions_maxs = self.policy.normalizer.normalizers["actions"].maxs
        actions_mins = self.policy.normalizer.normalizers["actions"].mins
        self.maxs = torch.tensor(np.concatenate([actions_maxs, observations_maxs]), device=self.device, dtype=torch.float)
        self.mins = torch.tensor(np.concatenate([actions_mins, observations_mins]), device=self.device, dtype=torch.float)
        def unnormalize(x):
            x = torch.clamp(x, -1, 1)
            x = (x + 1) / 2
            return x * (self.maxs - self.mins) + self.mins
        self.unnormalize = unnormalize

        self.is_setup = True
    
    def __call__(self, conditions, **kwargs):
        assert self.is_setup, "Policy is not setup. Please call setup() first."
        cond = self.policy._format_conditions(conditions, batch_size=self.args.per_sample_batch_size)
        x, extras = self.sample(cond, **kwargs)
        return self.tensor_to_obj(x), extras

    def sample(self, cond, **kwargs):
        x = self.diffusion(cond)
        extra_dict = {}
        return x, {}
        

    def tensor_to_obj(self, x):
        x = to_np(x)
        actions = x[..., :self.diffusion.action_dim]
        actions = self.policy.normalizer.unnormalize(actions, 'actions')
        action = actions[0,0]
        observations = x[..., self.diffusion.action_dim:]
        observations = self.policy.normalizer.unnormalize(observations, 'observations')
        
        trajectories = Trajectories(actions, observations)
        return action, trajectories


    