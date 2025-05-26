from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import torch

class FreedomGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(FreedomGuidance, self).__init__(args, **kwargs)

    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.guidance_strength * scheduler[t] * len(scheduler) / scheduler.sum()
    
    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        
        rho = self.get_rho(t, alpha_prod_ts, alpha_prod_t_prevs)
        
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t = ts[t]

        for recur_step in range(self.args.recur_steps):
            
            # we follow the exact algorithm in the paper
            
            # line 4 ~ 5
            epsilon = unet(x, t)
            x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)

            # line 6
            func = lambda zt: (zt - (1 - alpha_prod_t) ** (0.5) * unet(zt, t)) / (alpha_prod_t ** (0.5))

            # line 7
            guidance = self.guider.get_guidance(x.clone().detach().requires_grad_(), func, **kwargs)
            
            # line 8
            x_prev = x_prev + rho * guidance

            # line 9 ~ 11
            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
        
        print(x.abs().max().item())
        return x_prev