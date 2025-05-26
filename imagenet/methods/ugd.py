from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import torch

class UGDGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(UGDGuidance, self).__init__(args, **kwargs)

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

        for recur_step in range(self.args.recur_steps):
    
            # calculate hat z_0 as Equation (3)
            epsilon = unet(x, t)
            x0 = self._predict_x0(x, epsilon, alpha_prod_t, **kwargs)

            # calculate hat epsilon using forward universal guidance as Equation (6)
            x_need_grad = x.clone().detach().requires_grad_()
            # notice that ugd requires backpropagation within unet, so it's slow
            func = lambda zt: (zt - (1 - alpha_prod_t) ** (0.5) * unet(zt, t)) / (alpha_prod_t ** (0.5))
            guidance = self.guider.get_guidance(x_need_grad, func, **kwargs)
            # Here we follow the original paper to add sqrt(1 - alpha_t) into guidance strength
            epsilon_hat = epsilon - self.args.guidance_strength * guidance * ((1 - alpha_prod_t) ** (0.5))
            

            x0_for_gd = x0.clone().detach()
            # Here iter_steps - 1 is the "m" in the paper
            for iter_step in range(self.args.iter_steps - 1):
                # calculate hat z_0 using backward universal guidance as Equation (7)
                guidance = self.guider.get_guidance(x0_for_gd.requires_grad_(), **kwargs)
                x0_for_gd = x0_for_gd + self.args.guidance_strength * guidance
            
            delta_x0 = x0_for_gd - x0
            epsilon_hat = epsilon_hat - (alpha_prod_t / (1 - alpha_prod_t)) ** (0.5) * delta_x0

            # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
            x_prev = self._predict_x_prev_from_eps(x, epsilon_hat, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)

            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
        
        print(torch.max(x_prev.abs()).item())
        return x_prev