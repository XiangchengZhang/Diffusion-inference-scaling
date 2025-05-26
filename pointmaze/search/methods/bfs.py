import torch
from search.methods.base_guidance import BaseGuidance
from search.methods.tfg import TFGGuidance

from copy import deepcopy

class BFSGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.local_search = TFGGuidance(args, **kwargs)


    def get_temp(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.temp_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.temp_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.temp_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.temp * scheduler[t] * len(scheduler) / scheduler.sum()
    
    def evaluation_steps(self, **kwargs):
        start = self.args.start_step
        step = self.args.step_size
        end = self.args.inference_steps
        return list(range(start, end, step))

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
        
       


        x_prev, extra_results_dict = self.local_search.guide_step(
            x,
            i,
            unet, 
            ts,
            alpha_prod_ts,
            alpha_prod_t_prevs,
            eta,
            **kwargs,
        )
        x0 = extra_results_dict['x0']

        
        # allocate children according to verifier prob
        if self.args.temp > 0 and i in self.evaluation_steps():
            logprobs = self.guider.get_guidance(x0, return_logp=True, check_grad=False, **kwargs)
            logprobs = logprobs * self.get_temp(i, alpha_prod_ts, alpha_prod_t_prevs)
            probs = torch.softmax(logprobs, dim=0)
            
            num_children = probs * x.shape[0]
            num_children = torch.round(num_children).long()

            if self.args.method == 'bfs-resampling':
                resampled_indices = torch.arange(x.shape[0],device=x.device).repeat_interleave(num_children)
            elif self.args.method == 'bfs-pruning':
                resampled_indices = torch.where(num_children > 0)[0]
            elif self.args.method == 'bon':
                raise ValueError("Best-of-N must set temp = 0")
            else:
                raise NotImplementedError(f"Method {self.args.method} not implemented in BFS")
            
            x_prev = x_prev[resampled_indices]
            cond = kwargs['cond']
            for k, v in cond.items():
                if isinstance(v, torch.Tensor):
                    cond[k] = v[resampled_indices]
                else:
                    cond[k] = [v[i] for i in resampled_indices]
            


        # return the best sample at last step
        if i == len(ts) - 1:
            log_probs = self.guider.get_guidance(x_prev, return_logp=True, check_grad=False, **kwargs)
            index = torch.argmax(log_probs)
            x_prev = x_prev[index].unsqueeze(0)
            # print(log_probs[index].item())

        return x_prev, {}
