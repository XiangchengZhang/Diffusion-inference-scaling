import os
import numpy as np
from diffusers.utils import make_image_grid
from PIL import Image
from .base import BaseLogger

class ImageLogger(BaseLogger):

    def __init__(self, args, output_formats):
        
        super(ImageLogger, self).__init__(args, output_formats)
        self.image_size = args.image_size
        self.max_show_images = args.max_show_images
        self.logging_resolution = args.logging_resolution

    def _save_npy(self, data, fname):
        '''
            pack a list of PIL images into array.
        '''
        data = np.stack([np.array(d) for d in data])
        np.save(os.path.join(self.logging_dir, f"{fname}.npy"), data)
        self.log("save npy to %s" % os.path.join(self.logging_dir, f"{fname}.npy"))

    def _create_white_imgs(self, num):
        images = [Image.new('RGB', [self.image_size, self.image_size], (255, 255, 255)) for _ in range(num)]
        return images

    def log_samples(self, images, fname='images'):
        import math
        nrow = ncol = math.ceil(min(len(images), self.max_show_images) ** 0.5)
        
        # grid is a PIL image
        grid = make_image_grid(
            images[:nrow * ncol] + self._create_white_imgs(nrow*ncol-len(images)), rows=nrow, cols=ncol, resize=self.logging_resolution
        )
        
        # if we want to save fname under a subfolder
        if len(fname.split("/")) > 1:
            os.makedirs(os.path.join(self.logging_dir, os.path.dirname(fname)), exist_ok=True)

        self._save_npy(images, fname)

        self.log("Saving images to %s" % os.path.join(self.logging_dir, f"{fname}.png"))
        grid.save(os.path.join(self.logging_dir, f"{fname}.png"))

        if self.wandb:
            import wandb
            self.wandb_run.log({fname: wandb.Image(grid)})
        
        super(ImageLogger, self).log_samples(None)

    def load_samples(self, fname='images'):

        npy = np.load(os.path.join(self.logging_dir, f"{fname}.npy"))
        import json
        try:
            with open(os.path.join(self.logging_dir, f"metrics.json"), 'r') as f:
                metrics = json.load(f)
                compute = metrics.get('compute', 0)
        except (FileNotFoundError, json.JSONDecodeError):
            compute = 0
        if compute != 0:
            self.log(f"Loaded compute: {compute}")
        return [Image.fromarray(img) for img in npy], {'compute': compute}