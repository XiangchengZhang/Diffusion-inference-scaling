from .ddim import DDIMSampler
from diffusers.utils.torch_utils import randn_tensor
import math
from torch.autograd import grad
import torch
from functools import partial

from search.utils import rescale_grad



class TFGGuidance(DDIMSampler):

    def __init__(self, args, **kwargs):
        super(TFGGuidance, self).__init__(args, **kwargs)
        self.generator = torch.manual_seed(self.args.seed)
        def noise_fn (x, sigma, **kwargs):
            noise =  randn_tensor(x.shape, generator=self.generator, device=self.device, dtype=x.dtype)
            return sigma * noise + x
        self.noise_fn = noise_fn

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, return_logp=False, **kwargs):

        # flat_x0 = (x0[None] + mc_eps) #.reshape(-1, *x0.shape[1:])
        # v_func = torch.vmap(partial(self.guider.get_guidance,
        #                             return_logp=True, 
        #                             check_grad=False,
        #                             **kwargs))
        # outs = v_func(flat_x0)
        
        # avg_logprobs = torch.logsumexp(outs, dim=0) - math.log(mc_eps.shape[0])
        
        flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        outs = self.get_guidance(flat_x0, return_logp=True, check_grad=False, **kwargs)

        avg_logprobs = torch.logsumexp(outs.reshape(mc_eps.shape[0], x0.shape[0]), dim=0) - math.log(mc_eps.shape[0])
        
        if return_logp:
            return avg_logprobs

        _grad = torch.autograd.grad(avg_logprobs.sum(), x0)[0]
        _grad = rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad
    
    def get_noise(self, std, shape, eps_bsz=4, **kwargs):
        # if std == 0.0:
        #     return torch.zeros((1, *shape), device=self.device)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device), std, **kwargs) for _ in range(eps_bsz)]) 
    # randn_tensor((4, *shape), device=self.device, generator=self.generator) * std
    
    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.mu_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.mu *  scheduler[t] * len(scheduler) / scheduler.sum()
    
    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.sigma_schedule == 'decrease':    # beta_t
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.sigma *  scheduler[t]

    def guide_step(
        self,
        x: torch.Tensor,
        i: int,
        **kwargs,
    ) -> torch.Tensor:
        
        vec_s, vec_t = self.timesteps[i].expand(x.shape[0]), self.timesteps[i + 1].expand(x.shape[0])
        alpha_prod_ts = self.alpha_prod_ts
        alpha_prod_t_prevs = self.alpha_prod_t_prevs

        rho = self.get_rho(i, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self.get_mu(i, alpha_prod_ts, alpha_prod_t_prevs)
        std = self.get_std(i, alpha_prod_ts, alpha_prod_t_prevs)



        for recur_step in range(self.args.recur_steps):

            # sample noise to estimate the \tilde p distribution
            mc_eps = self.get_noise(std, x.shape, self.args.eps_bsz, **kwargs)

            # Compute guidance on x_t, and obtain Delta_t
            if self.args.rho != 0.0:
                with torch.enable_grad():
                    x_g = x.clone().detach().requires_grad_()
                    x_prev, x0 = self.solver_update(x_g, vec_s, vec_t)
                    x0 = x0['model_s']
                    logprobs = self.tilde_get_guidance(
                        x0, mc_eps, return_logp=True, **kwargs)
                    Delta_t = grad(logprobs.sum(), x_g)[0]
                    Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                    Delta_t = Delta_t * rho
                    
            else:
                Delta_t = torch.zeros_like(x)
                x_prev, x0 = self.solver_update(x, vec_s, vec_t)
                x0 = x0['model_s']
            # Compute guidance on x_{0|t}
            new_x0 = x0.clone().detach()
            
            for _ in range(self.args.iter_steps):
                if self.args.mu != 0.0:
                    new_x0 += mu * self.tilde_get_guidance(
                        new_x0.detach().requires_grad_(), mc_eps, **kwargs)
            Delta_0 = new_x0 - x0
            
            alpha_prod_t = alpha_prod_ts[i]
            alpha_prod_t_prev = alpha_prod_t_prevs[i]
            alpha_t = alpha_prod_t / alpha_prod_t_prev
            # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
            if i < self.args.inference_steps - 1 or recur_step < self.args.recur_steps - 1:
                x_prev += Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5
            x = self._predict_xt(x_prev, vec_t, vec_s).detach().requires_grad_(False)
        
        return x_prev




if __name__ == "__main__":
    pass
