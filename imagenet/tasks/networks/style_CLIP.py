import torch
from PIL import Image
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, ToTensor, Compose, Resize

class StyleCLIP(torch.nn.Module):

    def __init__(self, network, device, target=None):
        super(StyleCLIP, self).__init__()

        self.model = CLIPModel.from_pretrained(network)
        
        processor = AutoProcessor.from_pretrained(network).image_processor

        self.image_size = [processor.crop_size['height'], processor.crop_size['width']]

        self.transforms = Compose([
            Normalize(
                mean=processor.image_mean,
                std=processor.image_std
            ),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained(network)

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        if target is not None:
            self.target_embedding = self.get_target_embedding(target)

    @torch.no_grad()
    def get_target_embedding(self, target):
        img = Image.open(target).convert('RGB')
        image = img.resize(self.image_size, Image.Resampling.BILINEAR)
        image = self.transforms(ToTensor()(image)).unsqueeze(0)
        return self.get_gram_matrix(image)

    def get_gram_matrix(self, img):
        img = img.to(self.device)
        img = torch.nn.functional.interpolate(img, size=self.image_size, mode='bicubic')
        img = self.transforms(img)
        # following mpgd
        feats = self.model.vision_model(img, output_hidden_states=True, return_dict=True).hidden_states[2]        
        feats = feats[:, 1:, :]  # [bsz, seq_len, h_dim]
        gram = torch.bmm(feats.transpose(1, 2), feats)
        return gram

    def to_tensor(self, img):
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        return self.transforms(ToTensor()(img)).unsqueeze(0)

    def forward(self, x):

        embed = self.get_gram_matrix(x)
        diff = (embed - self.target_embedding).reshape(embed.shape[0], -1)
        similarity = -(diff ** 2).sum(dim=1).sqrt() / 100

        return similarity
    