import torch
import os
import scipy
from PIL import Image
from diffusers import AudioDiffusionPipeline
import numpy as np
from .utils import load_audio_dataset, check_grad_fn, rescale_grad


class AduioDeclippingGuidance:

    def __init__(self, args):

        self.args = args

        self.device = args.device

        self.images, self.audios = load_audio_dataset(args.dataset, args.num_samples)

        self._log_ref()

    def _log_ref(self):
        pipeline = AudioDiffusionPipeline.from_pretrained(self.args.model_name_or_path)

        def decode(images):
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")
            images = list(
                (Image.fromarray(_[:, :, 0]) for _ in images)
                if images.shape[3] == 1
                else (Image.fromarray(_, mode="RGB").convert("L") for _ in images)
            )
            audios = torch.tensor(np.array([pipeline.mel.image_to_audio(_) for _ in images]))
            return audios

                # saving here...
        mask = torch.ones_like(self.images)

        mask[:, :, :40, :] = 0
        mask[:, :, -40:, :] = 0
        
        corrupted_audios = self.images * mask

        ref_audio_dir = os.path.join(self.args.logging_dir, 'ref_audios')
        os.makedirs(ref_audio_dir, exist_ok=True)

        corrupted_audio_dir = os.path.join(self.args.logging_dir, 'corrupted_audios')
        os.makedirs(corrupted_audio_dir, exist_ok=True)

        ref_audios = decode(self.images) * 30            # * 30 means amplification
        corrupted_audios = decode(corrupted_audios) * 30

        for i, audio_tensor in enumerate(ref_audios):
            name = os.path.join(ref_audio_dir, f'audio_{i}.wav')
            scipy.io.wavfile.write(name, rate=self.args.sample_rate, data=audio_tensor.numpy())
        
        for i, audio_tensor in enumerate(corrupted_audios):
            name = os.path.join(corrupted_audio_dir, f'audio_{i}.wav')
            scipy.io.wavfile.write(name, rate=self.args.sample_rate, data=audio_tensor.numpy())
    
    @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x: x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):
        
        start_idx = self.args.batch_id * self.args.per_sample_batch_size
        end_idx = min((self.args.batch_id + 1) * self.args.per_sample_batch_size, len(self.images))

        if check_grad:
            check_grad_fn(x_need_grad)
        
        x = post_process(func(x_need_grad))

        mask = torch.ones_like(x)

        mask[:, :, :40, :] = 0
        mask[:, :, -40:, :] = 0

        noisy_images = self.images[start_idx:end_idx, ...].to(self.args.device) * mask

        # if mc is performed, noisy_images should have shape (mc_eps.shape[0], x0.shape[0], *x0.shape[1:])
        noisy_images = torch.cat([noisy_images for _ in range(x.shape[0] // noisy_images.shape[0])], dim=0)
        
        difference = noisy_images - x * mask

        # classifier returns log prob!
        log_probs = -torch.norm(difference, p=2, dim=(1, 2, 3))
        
        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)
        
    

