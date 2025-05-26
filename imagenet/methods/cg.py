from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import torch

class ClassifierGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(ClassifierGuidance, self).__init__(args, **kwargs)

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

        # alg 2 in classifier-guidance paper
        epsilon = unet(x, t)
        x_need_grad = x.clone().detach().requires_grad_(True)
        guidance = self.guider.get_guidance(x_need_grad, t, **kwargs)

        epsilon = epsilon - self.args.guidance_strength * guidance * ((1 - alpha_prod_t) ** (0.5))

        x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)

        return x_prev