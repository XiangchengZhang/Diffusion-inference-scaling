import torch
from typing import List, Union
from .utils import check_grad_fn, rescale_grad, ban_requires_grad

from .networks.resnet import ResNet18
from .networks.huggingface_classifier import HuggingfaceClassifier

from .networks.time_dependent_classifier import create_time_classifier

class ImageLabelGuidance:

    def __init__(self, guide_network, targets, device, time=False):
        self.guide_network = guide_network

        if isinstance(targets, str):
            targets = [int(x) for x in targets.split(',')]

        self._load_model(targets, time=time)

        self.classifier.to(device)
        self.classifier.eval()

        self.get_guidance = self.get_guidance_with_time if time else self.get_guidance_without_time

    def _load_model(self, y: Union[List[int], int], time: bool = False):

        # load resnet or huggingface model. Register more networks here.
        if time:
            self.classifier = create_time_classifier(target=y, guide_network=self.guide_network)
        
        else: 
            if 'resnet' in self.guide_network:
                self.classifier = ResNet18(targets=y, guide_network=self.guide_network)
            else:
                self.classifier = HuggingfaceClassifier(targets=y, guide_network=self.guide_network)
        
            ban_requires_grad(self.classifier)

    @torch.enable_grad()
    def get_guidance_with_time(self, x_need_grad, time=0, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

        if check_grad:
            check_grad_fn(x_need_grad)
        
        x_need_grad = func(x_need_grad)
        x = post_process(x_need_grad)

        log_probs = self.classifier(x, time, **kwargs)
        
        if return_logp:
            return log_probs
        
        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)

    @torch.enable_grad()
    def get_guidance_without_time(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

        if check_grad:
            check_grad_fn(x_need_grad)

        x = post_process(func(x_need_grad))
        
        # classifier returns log prob!
        log_probs = self.classifier(x)

        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)
        
    