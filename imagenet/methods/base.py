from diffusers.utils.torch_utils import randn_tensor
from tasks.base import BaseGuider
from utils.configs import Arguments
import torch

class BaseGuidance:

    def __init__(self, args: Arguments, noise_fn: None):

        self.args = args
        self.guider = BaseGuider(args)
        if noise_fn is None:
            self.generator = torch.manual_seed(self.args.seed)
            def noise_fn (x, sigma, **kwargs):
                noise =  randn_tensor(x.shape, generator=self.generator, device=self.args.device, dtype=x.dtype)
                return sigma * noise + x
            self.noise_fn = noise_fn
        else:
            self.noise_fn = noise_fn

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
    
            eps = unet(x, t)

            # predict x0 using xt and epsilon
            x0 = self._predict_x0(x, eps, alpha_prod_t, **kwargs)

            x_prev = self._predict_x_prev_from_zero(
                x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs
            )

            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
        
        return x_prev


    def _predict_x_prev_from_zero(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        eta: float,
        t: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        
        '''
            This function first compute (updated) eps from x_0, and then predicts x_{t-1} using Equation (12) in DDIM paper.
        '''
        
        new_epsilon = (
            (xt - alpha_prod_t ** (0.5) * x0) / (1 - alpha_prod_t) ** (0.5)
        )

        return self._predict_x_prev_from_eps(xt, new_epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)


    def _predict_x_prev_from_eps(
        self,
        xt: torch.Tensor,
        eps: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        eta: float,
        t: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        
        '''
            This function predicts x_{t-1} using Equation (12) in DDIM paper.
        '''

        sigma = eta * (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        ) ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - sigma**2) ** (0.5) * eps
        pred_x0_direction = (xt - (1 - alpha_prod_t) ** (0.5) * eps) / (alpha_prod_t ** (0.5))

        # Equation (12) in DDIM sampling
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_x0_direction + pred_sample_direction

        if eta > 0 and t.item() > 0:
            prev_sample = self.noise_fn(prev_sample, sigma, **kwargs)
            # variance_noise = randn_tensor(
            #     xt.shape, generator=self.generator, device=self.args.device, dtype=xt.dtype
            # )
            # variance = sigma * variance_noise

            # prev_sample = prev_sample + variance
        
        return prev_sample


    def _predict_xt(
        self,
        x_prev: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        
        xt_mean = (alpha_prod_t / alpha_prod_t_prev) ** (0.5) * x_prev

        return self.noise_fn(xt_mean, (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5), **kwargs)

        noise = randn_tensor(
            x_prev.shape, generator=self.generator, device=self.args.device, dtype=x_prev.dtype
        )   

        return xt_mean + (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5) * noise


    def _predict_x0(
        self, xt: torch.Tensor, eps: torch.Tensor, alpha_prod_t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        
        pred_x0 = (xt - (1 - alpha_prod_t) ** (0.5) * eps) / (alpha_prod_t ** (0.5))

        if self.args.clip_x0:
            pred_x0 = torch.clamp(pred_x0, -self.args.clip_sample_range, self.args.clip_sample_range)
        
        return pred_x0


