from diffusers import UNet2DModel, DDIMPipeline
import torch

def get_diffusion(args):
    pipeline = DDIMPipeline.from_pretrained(args.model_name_or_path)
    scheduler = pipeline.scheduler
    
    unet_ = pipeline.unet
    unet_.to(args.device)
    unet_.eval()
    unet = lambda x,t: unet_(x, t).sample

    scheduler.set_timesteps(args.inference_steps)
    ts = scheduler.timesteps

    alpha_prod_ts = scheduler.alphas_cumprod[ts]
    alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * scheduler.final_alpha_cumprod])

    return unet, ts, alpha_prod_ts, alpha_prod_t_prevs