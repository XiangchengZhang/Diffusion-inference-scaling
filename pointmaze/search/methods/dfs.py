from search.methods.tfg import TFGGuidance
from search.methods.base_guidance import BaseGuidance
from diffuser.models.helpers import apply_conditioning
from search.configs import Arguments
import torch
from typing import Tuple
from torch.autograd import grad
from search.utils import rescale_grad
import math


class DFSGuidance(BaseGuidance):
    def __init__(self, args:Arguments, **kwargs):
        super().__init__(args, **kwargs)
        self.local_search = TFGGuidance(args, **kwargs)
        self.reset()


    def get_threshold(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.threshold_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.threshold_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.threshold_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)
        return self.args.threshold / (scheduler[t] * len(scheduler) / scheduler.sum())  # to be tested

    def evaluation_steps(self, **kwargs):
        start = self.args.start_step
        step = self.args.step_size
        end = self.args.inference_steps
        return list(range(start, end, step))

    def reset(self, **kwargs):
        self.budget = self.args.budget
        self.buffer = [{} for _ in range(self.args.inference_steps)]

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
    ) -> Tuple[int, torch.Tensor]:
        
        x_prev, extra_results_dict = self.local_search.guide_step(x, i, unet, ts, alpha_prod_ts, alpha_prod_t_prevs, eta, **kwargs)
        x0 = extra_results_dict["x0"]
        logprobs = self.guider.get_guidance(x0, return_logp=True, check_grad=False, **kwargs)
        
        # if i == len(ts) - 1:
        #     print(logprobs.sum().item())
        
        accept = True
        if i in self.evaluation_steps():
            self.buffer[i][logprobs.sum().item()] = x_prev
            if -logprobs.sum() > self.get_threshold(i, alpha_prod_ts, alpha_prod_t_prevs) and self.budget > 0:
                accept = False
                self.budget -= 1
            elif -logprobs.sum() > self.get_threshold(i, alpha_prod_ts, alpha_prod_t_prevs) and self.budget == 0:
                accept = True
                x_prev = self.buffer[i][max(self.buffer[i].keys())] 
            else:
                accept = True
        
        if accept:
            return x_prev, {"i_next": i + 1}

        else:
            cond = kwargs['cond']
            next_noise_level = max(0, i - self.args.recur_depth)
            if next_noise_level == 0:
                x = torch.randn_like(x)
                x = apply_conditioning(x, cond, 2)
                return x, {"i_next": next_noise_level}
            alpha_prod_t = alpha_prod_ts[next_noise_level]
            alpha_prod_t_prev = alpha_prod_t_prevs[i]
            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs).detach().requires_grad_(False)
            x = apply_conditioning(x, cond, 2)
            return x, {"i_next": next_noise_level}
                
