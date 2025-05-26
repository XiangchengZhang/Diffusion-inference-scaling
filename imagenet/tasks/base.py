import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk, load_dataset
from diffusers import StableDiffusionPipeline
from functools import partial
import logger

from .image_label_guidance import ImageLabelGuidance
from .style_transfer_guidance import StyleTransferGuidance
from .super_resolution import SuperResolution
from .gaussian_deblur import GaussianDeblur
from .molecule_properties import MoleculePropertyGuidance
from .audio_declipping import AduioDeclippingGuidance
from .audio_inpainting import AduioInpaintingGuidance

class BaseGuider:

    def __init__(self, args):
        self.args = args
        self.generator = torch.manual_seed(args.seed)
        
        self.load_processor()   # e.g., vae for latent diffusion
        self.load_guider()      # guidance network

    def load_processor(self):
        if self.args.data_type == 'text2image':
            sd = StableDiffusionPipeline.from_pretrained(self.args.model_name_or_path)
            self.vae = sd.vae
            self.vae.eval()
            self.vae.to(self.args.device)
            for param in self.vae.parameters():
                param.requires_grad = False
            self.processor = lambda x: self.vae.decode(x / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
        else:
            self.processor = lambda x:  torch.clamp((x + 1) / 2, 0, 1)
            # self.processor = lambda x: x

    @torch.enable_grad()
    def process(self, x):
        return self.processor(x)

    @torch.no_grad()
    def load_guider(self):
        
        self.get_guidance = None

        # for combined guidance
        device = self.args.device

        guiders = []

        for task, guide_network, target in zip(self.args.tasks, self.args.guide_networks, self.args.targets):

            if task == 'style_transfer':
                guider = StyleTransferGuidance(guide_network, target, device)
            elif task == 'label_guidance':
                guider = ImageLabelGuidance(guide_network, target, device, time=False)
            elif task == 'label_guidance_time':
                guider = ImageLabelGuidance(guide_network, target, device, time=True)
            elif task == 'super_resolution':
                guider = SuperResolution(self.args)
            elif task == 'gaussian_deblur':
                guider = GaussianDeblur(self.args)
            elif task == 'molecule_property':
                guider = MoleculePropertyGuidance(self.args)
            elif task == 'audio_declipping':
                guider = AduioDeclippingGuidance(self.args)
            elif task == 'audio_inpainting':
                guider = AduioInpaintingGuidance(self.args)
            else:
                raise NotImplementedError
            
            guiders.append(guider)
        
        if len(guiders) == 1:
            self.get_guidance = partial(guider.get_guidance, post_process=self.process)
        else:
            self.get_guidance = partial(self._get_combined_guidance, guiders=guiders)

        if self.get_guidance is None:
            raise ValueError(f"Unknown guider: {self.args.guider}")
    
    def _get_combined_guidance(self, x, guiders, *args, **kwargs):
        values = []
        for guider in guiders:
            values.append(guider.get_guidance(x, post_process=self.process, *args, **kwargs))
        return sum(values)