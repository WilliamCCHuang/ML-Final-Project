import os
import random
import numpy as np

import torch


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


def get_feature_dim(encoder, img_size):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size)
        repr = encoder(x).view(1, -1)
        feature_dim = repr.shape[-1]

    return feature_dim


def get_augment_funcs(opt):
    if opt.training_scheme == 'supervised':
        augment_func = None

        return augment_func, None
    
    if opt.training_scheme == 'simclr':
        augment_func1 = None
        augment_func2 = None
        raise NotImplementedError
    
    if opt.training_scheme == 'byol':
        augment_func1 = None
        augment_func2 = None
        raise NotImplementedError


def compute_total_training_steps(loader, opt):
    return len(loader) * opt.epochs
