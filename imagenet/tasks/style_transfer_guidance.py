import torch

from .networks.style_CLIP import StyleCLIP
from .utils import check_grad_fn, rescale_grad, ban_requires_grad

class StyleTransferGuidance:

    def __init__(self, guide_network, target, device):
        
        # e.g., 'openai/clip-vit-base-patch16'
        self.guide_network = guide_network
        
        # an image path
        self.target = target
        
        self.device = device
        self._load_model()
    
    def _load_model(self):
        self.model = StyleCLIP(self.guide_network, self.device, target=self.target)

        ban_requires_grad(self.model)

    @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

        if check_grad:
            check_grad_fn(x_need_grad)
        
        x = post_process(func(x_need_grad))
        
        log_probs = self.model(x)

        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)