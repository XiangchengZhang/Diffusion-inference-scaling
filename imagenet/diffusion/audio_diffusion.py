import math
import torch
from tqdm import tqdm
from typing import List
import numpy as np
import os
from PIL import Image
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms.functional import to_tensor

from utils.configs import Arguments
from .base import BaseSampler
from methods.base import BaseGuidance
from diffusers import AudioDiffusionPipeline, DDIMScheduler
from utils.env_utils import *

class AudioDiffusionSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(AudioDiffusionSampler, self).__init__(args)
        
        self.audio_length_in_s = args.audio_length     # how long the audio is
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, pre v_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)

    @torch.no_grad()
    def decode(self, images):

        if self.pipeline.vqvae is not None:
            images = 1 / self.pipeline.vqvae.config.scaling_factor * images
            images = self.pipeline.vqvae.decode(images)["sample"]
        
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = list(
            (Image.fromarray(_[:, :, 0]) for _ in images)
            if images.shape[3] == 1
            else (Image.fromarray(_, mode="RGB").convert("L") for _ in images)
        )

        audios = torch.tensor(np.array([self.pipeline.mel.image_to_audio(_) for _ in images]))
        return audios

    @torch.no_grad()
    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''

        self.pipeline = AudioDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.pipeline.scheduler.config.num_train_timesteps,
            beta_start=self.pipeline.scheduler.config.beta_start,
            beta_end=self.pipeline.scheduler.config.beta_end,
            beta_schedule=self.pipeline.scheduler.config.beta_schedule
        )

        self.pipeline.scheduler = self.scheduler
        
        self.args.sample_rate = self.pipeline.mel.sample_rate

        unet = self.pipeline.unet
        if isinstance(unet.config.sample_size, int):
            unet.config.sample_size = (unet.config.sample_size, unet.config.sample_size)
        
        unet.eval()

        for param in unet.parameters():
            param.requires_grad = False

        self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

        self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = unet, ts, alpha_prod_ts, alpha_prod_t_prevs

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):
        
        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

        for batch_id in range(n_batchs):
            
            self.args.batch_id = batch_id

            noise = randn_tensor(
                (
                    self.per_sample_batch_size,
                    self.pipeline.unet.config.in_channels,
                    self.pipeline.unet.config.sample_size[0],
                    self.pipeline.unet.config.sample_size[1],
                ),
                generator=self.generator,
                device=self.device,
            )
            images = noise

            for t in tqdm(range(self.inference_steps), total=self.inference_steps):
                
                def audio_diffusion_unet(x, t):
                    return self.pipeline.unet(x, t)['sample']

                images = guidance.guide_step(
                    images, t, audio_diffusion_unet,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta
                )

            audio = self.decode(images)
            tot_samples.append(audio.clone().cpu())
        
        return torch.concat(tot_samples)
    
    def tensor_to_obj(self, x: torch.Tensor):
        return [(x * self.args.volume_factor, self.args.sample_rate) for i, x in enumerate(x.numpy())]

    def obj_to_tensor(objs: List) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        audios = [torch.tensor(obj[0]) for obj in objs]
        tensor_audios = torch.stack(audios)
        return tensor_audios