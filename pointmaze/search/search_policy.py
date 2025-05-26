from search.configs import Arguments
from search.base_policy import BasePolicy
from search.methods.base_guidance import BaseGuidance
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffuser.models.helpers import apply_conditioning

class SearchPolicy(BasePolicy):
    def __init__(self, args:Arguments, **kwargs):
        super().__init__(args=args, **kwargs)
    
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        self.inference_steps = self.args.inference_steps
        self.eta = self.args.eta
        self.generator = torch.manual_seed(self.args.seed)
        self.per_sample_batch_size = self.args.per_sample_batch_size

        from search.utils import space_timesteps

        ts = space_timesteps(self.diffusion.n_timesteps, self.inference_steps) ## reversed order
        alpha_prod_ts = self.diffusion.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat((self.diffusion.alphas_cumprod[ts[1:]], torch.tensor([1.0], device=self.diffusion.alphas_cumprod.device, dtype=torch.float32)),dim=0)

        ## shape: [inference_steps]
        self.ts = torch.tensor(ts, device=self.device, dtype=torch.long)
        self.alpha_prod_ts = alpha_prod_ts
        self.alpha_prod_t_prevs = alpha_prod_t_prevs


    def sample(self, cond, guidance:BaseGuidance, **kwargs):
        x = randn_tensor((self.per_sample_batch_size, self.diffusion.horizon,self.diffusion.transition_dim), generator=self.generator, device=self.device)
        x = apply_conditioning(x, cond, self.diffusion.action_dim)
        guidance.reset()
        i = 0
        total_compute = 0
        while i < len(self.ts):
            total_compute += self.args.recur_steps * x.shape[0]   ## NFEs per step
            x, extras = guidance.guide_step(
                x,
                i,
                self.diffusion.model,
                self.ts,
                self.alpha_prod_ts,
                self.alpha_prod_t_prevs,
                self.eta,
                cond=cond,
                post_process=self.unnormalize
            )

            i = extras.get("i_next", i + 1)
          
        return x, {'compute': total_compute}

    

    
    




