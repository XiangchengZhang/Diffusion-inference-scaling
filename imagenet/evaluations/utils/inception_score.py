import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3, Inception_V3_Weights
from torchvision.transforms.transforms import Resize, Normalize, ToTensor, Compose
import numpy as np
from scipy.stats import entropy

class ImagePILDataset(torch.utils.data.Dataset):
    def __init__(self, images, transforms=None):
        self.files = images
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = self.files[i]
        img = image.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

@torch.no_grad()
def inception_score(imgs, device, batch_size=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    dataset = ImagePILDataset(imgs, transforms=Compose(
        [ToTensor(), Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
    ))

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)

    # Load inception model
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear')
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    preds = []

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        preds.append(get_pred(batch))
    
    preds = np.concatenate(preds, 0)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)
