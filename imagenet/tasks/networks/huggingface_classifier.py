import torch
from transformers import AutoModelForImageClassification, AutoProcessor
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, Resize

def convert_age_logp(logp: torch.Tensor):
    newlogp = torch.concat([logp[:, :3].mean(dim=1, keepdim=True), logp[:, 5:].mean(dim=1, keepdim=True)], dim=1)
    return newlogp

class HuggingfaceClassifier(nn.Module):

    def __init__(self, targets, guide_network=None):
        
        super(HuggingfaceClassifier, self).__init__()
        
        self.model = AutoModelForImageClassification.from_pretrained(guide_network)
        
        processor = AutoProcessor.from_pretrained(guide_network)
        
        self.transforms = Compose([
            Resize([processor.size['height'], processor.size['width']]),
            Normalize(mean=processor.image_mean, std=processor.image_std)
        ])
        
        self.target = targets

        self.model.eval()

    @torch.enable_grad()
    def forward(self, x):
        '''
            return tensor in the shape of (batch_size, )
        '''
        
        target = self.target

        resized_x = self.transforms(x)
        logits = self.model(resized_x).logits

        probs = torch.nn.functional.softmax(logits, dim=1)
        
        log_probs = torch.log(probs)
        
        if isinstance(target, int) or isinstance(target, str):
            selected = log_probs[range(x.size(0)), int(target)]
        else:
            selected = torch.cat([log_probs[range(x.size(0)), _] for _ in target], dim=0)

        return selected


    

        