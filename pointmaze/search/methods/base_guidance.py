from diffusers.utils.torch_utils import randn_tensor
from search.configs import Arguments
import torch
from diffuser.models.helpers import apply_conditioning
from search.maze_verifier import MazeVerifier
from search.utils import rescale_grad

class BaseGuidance:

    def __init__(self, args: Arguments, noise_fn: None=None):

        self.args = args
        self.guider = MazeVerifier(args)
        self.generator = torch.manual_seed(self.args.seed)
        self.device = torch.device(self.args.device)
        if noise_fn is None:
            def noise_fn (x, sigma, **kwargs):
                noise =  randn_tensor(x.shape, generator=self.generator, device=self.device, dtype=x.dtype)
                return sigma * noise + x
            self.noise_fn = noise_fn
        else:
            self.noise_fn = noise_fn

    def reset(self, **kwargs):
        pass
    
    def tilde_get_guidance(self, x0, return_logp=False, check_grad=True, **kwargs):
        outs = self.guider.get_guidance(x=x0, return_logp=return_logp, check_grad=check_grad, **kwargs)
        avg_logprobs = outs
        if return_logp:
            return avg_logprobs
        _grad = torch.autograd.grad(avg_logprobs.sum(), x0)[0]
        _grad = rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad



    def guide_step(
        self,
        x: torch.Tensor,
        i: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:


        cond = kwargs.pop("cond", None)
        t = ts[i]
        alpha_prod_t = alpha_prod_ts[i]
        alpha_prod_t_prev = alpha_prod_t_prevs[i]

        batched_t = t.repeat(x.shape[0])
        
        eps = unet(x, cond, batched_t)

        # predict x0 using xt and epsilon
        x0 = self._predict_x0(x, eps, alpha_prod_t, **kwargs)

        x_prev = self._predict_x_prev_from_zero(
            x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs
        )
        x_prev = apply_conditioning(x_prev, cond, action_dim=2)

        # x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
        # x = apply_conditioning(x, cond, action_dim=2)
        
        if t == 0:   ## Do best-of-n sampling at final step
            log_probs = self.tilde_get_guidance(x_prev, return_logp=True, check_grad=False, **kwargs)
            index = log_probs.argmax(dim=0)
            x_prev = x_prev[index].unsqueeze(0)

        return x_prev, {}


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
        
        # pred_x0 = (xt - (1 - alpha_prod_t) ** (0.5) * eps) / (alpha_prod_t ** (0.5))

        # if self.args.clip_x0:
        #     pred_x0 = torch.clamp(pred_x0, -self.args.clip_sample_range, self.args.clip_sample_range)
        
        return eps


