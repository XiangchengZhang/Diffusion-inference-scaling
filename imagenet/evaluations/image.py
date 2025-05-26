import multiprocessing as mp
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torchvision import transforms
from torch.utils.data import DataLoader
from .utils.fid import calculate_fid
from .utils.inception_score import inception_score
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore

from .base import BaseEvaluator
from tasks.utils import load_image_dataset
from utils.env_utils import *
import logger

class ImageEvaluator(BaseEvaluator):

    def __init__(self, args):
        super(ImageEvaluator, self).__init__()

        self.args = args
    
    @torch.no_grad()
    def _compute_validity(self, images, labels, return_preds=False, guide_networks=None):
        
        if guide_networks is None:
            guide_networks = self.args.guide_networks
        
        if not isinstance(guide_networks, list):
            guide_networks = [guide_networks]
        
        if not isinstance(labels, list):
            labels = [labels]
        
        correct = torch.zeros(len(images))
        validities = []

        for label, guide_network in zip([int(l) for l in labels], guide_networks):
            model_path_or_model = COND_VALIDITY_PATH_MAPPING[guide_network]
            self.feature_extractor = AutoImageProcessor.from_pretrained(model_path_or_model)
            model = AutoModelForImageClassification.from_pretrained(model_path_or_model)
            model.eval()
            self.classifier = model
            self.classifier.to(self.args.device)

            preds = self._get_prediction(images, batchsize=self.args.eval_batch_size, guide_network=guide_network)

            if return_preds:
                return preds
            
            correct_partial = (torch.tensor(preds) == torch.tensor(label)).float()
            correct += correct_partial
            validity_partial = correct_partial.mean().item()

            validities.append(validity_partial)

        validity = correct.mean().item() / len(guide_networks)
        return validity, validities

    @torch.no_grad()
    def _compute_lpips(self, images):
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(self.args.device)
        
        target_img = load_image_dataset(self.args.dataset, num_samples=self.args.num_samples, target=self.args.targets, return_tensor=True, normalize=False).to(self.args.device)

        sample_img = torch.stack([transforms.ToTensor()(img) for img in images], dim=0).to(self.args.device)
        lpips_score = lpips(sample_img, target_img)
        return lpips_score.item()

    @torch.no_grad()
    def _compute_kid(self, images):

        kid = KernelInceptionDistance(subset_size=min(len(images), 50), normalize=True).to(self.args.device)
        
        def image_proc(image):
            tf = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])
            return torch.stack([tf(img) for img in image], dim=0).to(self.args.device)

        ref_images = load_image_dataset(self.args.dataset, num_samples=1000, target=-1, return_tensor=False)

        for i in range(0, len(ref_images), self.args.eval_batch_size):
            ref_batch = ref_images[i:i+self.args.eval_batch_size]
            ref_batch = image_proc(ref_batch)
            kid.update(ref_batch, real=True)
       
        for i in range(0, len(images), self.args.eval_batch_size):
            batch = images[i:i+self.args.eval_batch_size]
            batch = image_proc(batch)
            kid.update(batch, real=False)
        
        score = kid.compute()
        return score[0].item()

        
    @torch.no_grad()
    def _get_prediction(self, samples, batchsize=None, guide_network=None):
        assert self.feature_extractor is not None
        assert self.classifier is not None

        if guide_network is None:
            guide_network = self.args.guide_network

        if batchsize is None:
            batchsize = len(samples)
        
        # iterate over the samples with batch size
        for i in range(0, len(samples), batchsize):
            inputs = self.feature_extractor(samples[i:i+batchsize], return_tensors="pt")
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            outputs = self.classifier(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # this is a hard coding for the specific models
            if 'age' in guide_network:
                probs = torch.cat([probs[:, :3].mean(dim=1, keepdim=True), probs[:, 5:].mean(dim=1, keepdim=True)], dim=1)
            elif 'hair' in guide_network:
                probs = torch.cat([probs[:, 2:3], probs[:, 3:4], probs[:, 0:1], probs[:, 1:2]], dim=1)

            if i == 0:
                all_probs = probs
            else:
                all_probs = torch.cat([all_probs, probs], dim=0)
        
        return all_probs.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def _compute_clip_score(self, images, text):

        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(self.args.device)
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
        )

        for bs in range(0, len(images), self.args.eval_batch_size):
            metric.update(
                images=[tf(img).to(self.args.device) for img in images[bs:bs+self.args.eval_batch_size]],
                text=text[bs: bs+self.args.eval_batch_size],
            )

        return metric.compute().item()
    
    @torch.no_grad()
    def _compute_style_score(self, images, targets):
        from tasks.networks.style_CLIP import StyleCLIP
        clip_embedding = StyleCLIP(device=self.args.device, network='openai/clip-vit-base-patch32')

        similarity_list = []

        for target in targets:

            target_embed = clip_embedding.get_target_embedding(target).to(self.args.device)
            image_embed = []
            
            for bs in range(0, len(images), self.args.eval_batch_size):
                imgs = [clip_embedding.to_tensor(img) for img in images[bs:bs+self.args.eval_batch_size]]
                image_embed.append(clip_embedding.get_gram_matrix(torch.concat(imgs, dim=0).to(self.args.device)))
            
            image_embed = torch.concat(image_embed, dim=0)
            diff = (image_embed - target_embed).reshape(image_embed.size(0), -1)
            similarity_list.append(-(diff ** 2).sum(dim=1).sqrt() / 10)

        return torch.cat(similarity_list, dim=0).mean().item() / 10.0

    @torch.no_grad()
    def _compute_fid(self, samples, dataset, target):

        # ref_images = load_image_dataset(dataset, num_samples=-1, target=target, return_tensor=False)
        ref_images = f"stats/imagenet-1k-fid-stats-{str(target[0])}.pt"

        fid = calculate_fid(ref_images, samples, self.args.eval_batch_size, self.args.device)

        return fid
    
    def _compute_inception_score(self, samples):
        scores = inception_score(samples, self.args.device, batch_size=self.args.eval_batch_size)
        return scores
     
    def evaluate(self, samples):
        metrics = {}
        logger.log(f"Evaluating {len(samples)} samples")
        
        inception_score = self._compute_inception_score(samples)
        metrics['inception_score'] = inception_score

        # we only allow combined guidance within the same dataset
        if self.args.dataset in ['imagenet', 'cifar10', 'cat']:
            fid = self._compute_fid(samples, self.args.dataset, self.args.targets)
            metrics['fid'] = fid
            # print("skip fid")

        
        if self.args.dataset in ['celebahq']:
            kid = self._compute_kid(samples)
            metrics['log_kid'] = np.log(kid)

        if 'label_guidance' in self.args.tasks or 'label_guidance_time' in self.args.tasks:
            validity, validities = self._compute_validity(samples, self.args.targets)
            metrics['validity'] = validity

            for i, guide_network in enumerate(self.args.guide_networks):
                metrics[f'validity_{guide_network}'] = validities[i]

        if 'super_resolution' in self.args.tasks or 'gaussian_deblur' in self.args.tasks or 'phase_retrieval' in self.args.tasks:
            lpips = self._compute_lpips(samples)
            metrics['neg_lpips'] = -lpips
        
        if 'style_transfer' in self.args.tasks:
            style_score = self._compute_style_score(samples, self.args.targets)
            metrics['style_score'] = style_score
            clip_score = self._compute_clip_score(samples, self.args.targets)
            metrics['clip_score'] = clip_score

        return metrics
