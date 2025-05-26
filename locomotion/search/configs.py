import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, List

@dataclass
class Arguments:
    
    # data related
    data_type: Literal['traj'] = field(default='traj')
    dataset: str = field(default='halfcheetah-medium-expert-v2')
    task: List[int] = field(default_factory=lambda: [1,])
    method: str = field(default='tfg')   

    # diffusion related
    train_steps: int = field(default=1000)
    inference_steps: int = field(default=15)
    eta: float = field(default=1.0)
    clip_x0: bool = field(default=True)
    clip_sample_range: float = field(default=1.0)

    # inference related:
    seed: int = field(default=3)
    device: str = field(default='cuda')
    logging_dir: str = field(default='logs')
    per_sample_batch_size: int = field(default=1)
    num_samples: int = field(default=10)
    batch_id: int = field(default=0)    # start from the zero

    # guidance related
    guidance_name: str = field(default='no')
    recur_steps: int = field(default=1)    
    iter_steps: int = field(default=1)
    guidance_strength: float = field(default=0.0)  # guidance scale for time dependent guidance in basline
    clip_scale: float = field(default=100)

    # specific for local search
    rho: float = field(default=0.04)
    mu: float = field(default=0.01)
    sigma: float = field(default=0.00)
    eps_bsz: int = field(default=1)
    rho_schedule: str = field(default='increase')
    mu_schedule: str = field(default='increase')
    sigma_schedule: str = field(default='decrease')

    