import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from byol import BYOL
from datasets import ImagenetteDataset
from resnet import get_resnet
from utils import seed_everything, get_feature_dim, get_transform, compute_total_training_steps


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-dir', type=str)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--training-scheme', type=str, default='byol')
    parser.add_argument('--num-layers', type=int, default=50)
    parser.add_argument('--wide-scale', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)  # 4096 in paper
    parser.add_argument('--lr-base', type=float, default=0.2)
    parser.add_argument('--tau-base', type=float, default=0.996)

    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='seed')

    opt = parser.parse_args()

    return opt


def check_opt(opt):
    opt.training_scheme in ['supervised', 'simclr', 'byol']
    opt.num_layers in [50, 101, 152, 200]
    opt.wide_scale in [1, 2, 3, 4]

    if 'imagenette' in opt.img_dir:
        opt.num_classes = 10

    opt.lr = opt.lr_base * opt.batch_size / 256


def main():
    opt = parse_opt()
    check_opt(opt)
    seed_everything(opt.seed)
    
    device = torch.device(f'cuda:{opt.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    encoder = get_resnet(opt).to(device)
    opt.feature_dim = get_feature_dim(encoder, opt.img_size, device)

    scheme_func = None
    if opt.training_scheme == 'supervised':
        scheme_func = train_supervised
    elif opt.training_scheme == 'simclr':
        scheme_func = train_simclr
    else:
        scheme_func = train_byol

    opt.augment_funcs = get_transform(opt)
    scheme_func(encoder, opt, device)


def train_supervised(encoder, opt, device):
    raise NotImplementedError  # TODO:


def train_simclr(loader, encoder, opt, device):
    raise NotImplementedError  # TODO:


def train_byol(encoder, opt, device):
    train_loader, val_loader = get_loaders(opt)
    transform = get_transform(opt)
    total_training_steps = compute_total_training_steps(train_loader, opt)

    learner = BYOL(
        encoder=encoder,
        feature_dim=opt.feature_dim,
        augment_func1=transform,
        augment_func2=transform,
        tau_base=opt.tau_base,
        total_training_steps=total_training_steps,
    )

    optimizer = optim.Adam(learner.parameters(), lr=opt.lr)

    t_epoch = tqdm(range(opt.epochs), desc='Epochs')
    for epoch in t_epoch:
        t_batch = tqdm(train_loader)
        for i, (img, _) in enumerate(t_batch):
            current_training_steps = epoch * len(train_loader) + i

            loss = learner(img.to(device))
            optimizer.zero_grad()
            loss.backward()
            learner.update_target_network(current_training_steps=current_training_steps)

            t_batch.set_postfix({'byol loss': f'{loss.item():.4f}'})

        with torch.no_grad():
            losses = []
            for img, _ in enumerate(val_loader):
                loss = learner(img.to(device))
                losses.append(loss.item())

        loss = np.mean(losses)
        t_epoch.set_postfix({'byol loss': f'{loss:.4f}'})


def get_loaders(opt):
    train_dataset = ImagenetteDataset(Path(opt.img_dir) / 'train', opt.img_size)
    val_dataset = ImagenetteDataset(Path(opt.img_dir) / 'val', opt.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    return train_loader, val_loader
    

if __name__ == '__main__':
    main()
