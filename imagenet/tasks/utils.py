import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torchvision import transforms
from diffusers import AudioDiffusionPipeline
from torchvision.transforms.functional import to_tensor

from utils.env_utils import *

def ban_requires_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def check_grad_fn(x_need_grad):
    assert x_need_grad.requires_grad, "x_need_grad should require grad"


def rescale_grad(
    grad: torch.Tensor, clip_scale, **kwargs
):  # [B, N, 3+5]
    node_mask = kwargs.get('node_mask', None)

    scale = (grad ** 2).mean(dim=-1)
    if node_mask is not None:  # [B, N, 1]
        scale: torch.Tensor = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)  # [B]
        clipped_scale = torch.clamp(scale, max=clip_scale)
        co_ef = clipped_scale / scale  # [B]
        grad = grad * co_ef.view(-1, 1, 1)

    return grad

def load_audio_dataset(dataset, num_samples=-1):
    
    if dataset == 'teticio/audio-diffusion-256':

        dataset = load_dataset('teticio/audio-diffusion-256')
        dataset = dataset.shuffle(seed=42)
        
        images = dataset['train'][:num_samples]['image']
        
        pipeline = AudioDiffusionPipeline.from_pretrained('teticio/audio-diffusion-256')
        
        audios = torch.tensor(np.array([pipeline.mel.image_to_audio(image) for image in images]))
        images = torch.stack([to_tensor(_) for _ in images])
        normalized_images = (images - 0.5) / 0.5

    else:
        raise NotImplementedError

    return normalized_images, audios
        

def load_image_dataset(dataset, num_samples=-1, target=-1, return_tensor=True, normalize=True):

    if dataset == 'cat':
        images = load_dataset("cats_vs_dogs")
        images = images.filter(lambda x: x == 0, input_columns='labels')
        images = images['train'][:num_samples]['image']
        if not return_tensor:
            images = [images.resize((256, 256)) for images in images]
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    
    elif dataset == 'cifar10':
        dataset = load_dataset('cifar10')
        
        if target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')
        
        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('img', 'images')
        dataset = concatenate_datasets([dataset['train'], dataset['test']])
        
        if num_samples > 0:
            dataset = dataset[:num_samples]

        images = [images.resize((32, 32)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    elif dataset == 'imagenet':
        dataset = load_from_disk(IMAGENET_PATH)

        if target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')
        
        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('image', 'images')
        
        if num_samples > 0:
            dataset = dataset[:num_samples]
        
        images = [images.resize((256, 256)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif dataset == 'celebahq':
        images = []
        
        if num_samples < 0:
            num_samples = len(os.listdir(CELEBA_PATH))
        
        for img in os.listdir(CELEBA_PATH)[:num_samples]:
            images.append(Image.open(os.path.join(CELEBA_PATH, img)))
        
        tf = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])

    elif dataset == 'bird-species':
        
        dataset = load_dataset('chriamue/bird-species-dataset')
        dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
        
        if target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')
        
        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('image', 'images')
        
        if num_samples > 0:
            dataset = dataset[:num_samples]
        
        images = [images.resize((256, 256)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    else:
        raise NotImplementedError

    if normalize:
        tf.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if return_tensor:
        image_tensors = [tf(img) for img in images]
        return torch.stack(image_tensors, dim=0)
    else:
        return images