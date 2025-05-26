from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import torch

class DPSGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(DPSGuidance, self).__init__(args, **kwargs)

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
        
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t = ts[t]

        epsilon = unet(x, t)
        x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)

        func = lambda zt: (zt - (1 - alpha_prod_t) ** (0.5) * unet(zt, t)) / (alpha_prod_t ** (0.5))

        guidance = self.guider.get_guidance(x.clone().detach().requires_grad_(), func, **kwargs)
        
        # follow the schedule of DPS paper
        logp_norm = self.guider.get_guidance(x.clone().detach(), func, return_logp=True, check_grad=False, **kwargs)
        
        x_prev = x_prev + self.args.guidance_strength * guidance / torch.abs(logp_norm.view(-1, * ([1] * (len(x_prev.shape) - 1)) ))

        return x_prev