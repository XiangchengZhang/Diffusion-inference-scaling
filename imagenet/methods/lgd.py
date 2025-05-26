from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import torch
from tasks.utils import rescale_grad


class LGDGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(LGDGuidance, self).__init__(args, **kwargs)

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

        # the method name is like "lgd_10"
        eps_btz = int(self.args.guidance_name.split("_")[-1])

        # (eps_btz, bsz, c, h, w)
        mc_eps = torch.stack([self.noise_fn(torch.zeros_like(x), (1-alpha_prod_t)**0.5, **kwargs) for _ in range(eps_btz)])
        # mc_eps = randn_tensor((eps_btz, *x.shape), device=x.device, generator=self.generator) * (1 - alpha_prod_t) ** (0.5)

        # guidance function (we want gradient on xt)
        def func(xt): 
            predict_x0 = (xt - (1 - alpha_prod_t) ** (0.5) * unet(xt, t)) / (alpha_prod_t ** (0.5)) # (bsz, c, h, w)
            # add variance to the predict_x0
            predict_x0 = predict_x0[None] + mc_eps # (eps_btz, bsz, c, h, w)
            # flatten the predict_x0 for logp computation
            flatten_predict_x0 = predict_x0.reshape(-1, *x.shape[1:]) # (eps_btz * bsz, c, h, w)
            return flatten_predict_x0

        with torch.enable_grad():
            # flatten_log_prob: (eps_btz * bsz)
            x_need_grad = x.clone().detach().requires_grad_()
            x0_grad = (x_need_grad - (1 - alpha_prod_t) ** (0.5) * unet(x_need_grad, t)) / (alpha_prod_t ** (0.5))
            flat_x0 = x0_grad[None] + mc_eps
            from functools import partial
            v_func = torch.vmap(partial(self.guider.get_guidance,
                            return_logp=True, 
                            check_grad=False,
                            **kwargs))
    
            # flatten_log_prob = self.guider.get_guidance(x_need_grad, func, return_zero_order=True, **kwargs)
            # split into (eps_btz, bsz)
            log_prob = v_func(flat_x0)
            # log_prob = flatten_log_prob.reshape(eps_btz, -1)
            # compute logsumexp over eps_btz: log(1/n * sum(exp(log_prob))) = log(1/n) + logsumexp(log_prob)
            # since we only need to get the gradient of log_prob, we can ignore the constant term log(1/n)
            log_prob = torch.logsumexp(log_prob, dim=0)
            # divide by eps_btz to get the expectation
            guidance = torch.autograd.grad(log_prob.sum(), x_need_grad)[0] / eps_btz
            guidance = rescale_grad(guidance, clip_scale=self.args.clip_scale, **kwargs)


        x_prev = x_prev + self.args.guidance_strength * guidance

        return x_prev