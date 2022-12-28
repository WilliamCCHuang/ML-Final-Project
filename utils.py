import os
import random
import numpy as np

import torch
from torchvision.transforms import transforms


def check_opt(opt):
    raise NotImplementedError


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_feature_dim(encoder, img_size, device):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size).to(device)
        repr = encoder(x).view(1, -1)
        feature_dim = repr.shape[-1]

    return feature_dim


def get_transform(opt):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = [
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
        # transforms.RandomApply([Solarize()], p=solarize_prob),
        # transforms.ToTensor(),
        normalize
    ]
    
    return transforms.Compose(transform)


def compute_total_training_steps(loader, opt):
    return len(loader) * opt.epochs
