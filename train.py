import copy
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from byol import BYOL
from simclr import SimCLR
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--tau-base', type=float, default=0.996)
    parser.add_argument('--temperature', type=float, default=0.5)

    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='seed')

    opt = parser.parse_args()

    return opt


def check_opt(opt):
    opt.training_scheme in ['supervised', 'simclr', 'byol']
    if opt.training_scheme in ['supervised']:
        opt.last_fc = True
    elif opt.training_scheme in ['simclr', 'byol']:
        opt.last_fc = False

    assert opt.num_layers in [18, 50, 101, 152, 200]
    assert opt.wide_scale in [1, 2, 3, 4]

    if 'imagenette' in opt.img_dir:
        opt.num_classes = 10

    if opt.output_dir is None:
        raise RuntimeError('The argument `output-dir` must be assign')
    opt.output_dir = Path(opt.output_dir)
    if not opt.output_dir.exists():
        opt.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('Config:')
    pprint(vars(opt))
    print()


def save_opt(opt):
    opt_path = str(opt.output_dir / 'config.yaml')

    cfg = copy.deepcopy(vars(opt))
    for k, v in cfg.items():
        if isinstance(v, Path):
            cfg[k] = str(v)

    with open(str(opt_path), 'w') as f:
        yaml.dump(cfg, f)


def main():
    opt = parse_opt()
    check_opt(opt)
    save_opt(opt)
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

    scheme_func(encoder, opt, device)


def train_supervised(model, opt, device):
    train_loader, val_loader = get_loaders(opt)
    train_loader.dataset.transform_2 = None
    val_loader.dataset.transform_2 = None

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    best_acc = 0
    t_epoch = tqdm(range(opt.epochs), desc='Epochs')
    for epoch in t_epoch:
        t_batch = tqdm(train_loader, desc='Batches')
        for i, (img, _, label) in zip(t_batch):
            img = img.to(device)
            label = label.to(device)

            y_prob = model(img)
            loss = loss_fn(y_prob, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / len(train_loader))

            t_batch.set_postfix({'train loss': f'{loss.item():.4f}'})

        with torch.no_grad():
            val_acc = 0
            val_loss = []
            for img, _, label in val_loader:
                img = img.to(device)
                label = label.to(device)

                y_prob = model(img)
                loss = loss_fn(y_prob, label)
                val_loss.append(loss.item())

                _, y_pred = torch.max(y_prob, dim=1)
                val_acc += (label == y_pred).sum()

        val_loss = np.mean(val_loss)
        val_acc = val_acc.item() / len(val_loader.dataset)
        t_epoch.set_postfix({'val loss': f'{val_loss:.4f}', 'val acc': f'{val_acc:.4f}'})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(opt.output_dir / 'model.pt'))
            print(f'save model as val acc = {val_acc:.4f}')


def train_simclr(encoder, opt, device):
    train_loader, val_loader = get_loaders(opt)
    transform = get_transform(opt)

    learner = SimCLR(
        encoder=encoder,
        transform_1=transform,
        transform_2=transform,
        feature_dim=opt.feature_dim,
        temperature=opt.temperature
    )

    optimizer = optim.Adam(learner.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    best_loss = np.inf
    t_epoch = tqdm(range(opt.epochs), desc='Epochs')
    for _ in t_epoch:
        t_batch = tqdm(train_loader, desc='Batches')
        for img, _ in t_batch:
            loss = learner(img.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_batch.set_postfix({'train loss': f'{loss.item():.4f}'})

        with torch.no_grad():
            val_loss = []
            for img, _ in val_loader:
                loss = learner(img.to(device))
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        t_epoch.set_postfix({'val loss': f'{val_loss:.4f}'})

        if val_loss < best_loss:
            best_loss = val_loss
            learner.save(dir_path=opt.output_dir)
            print(f'save model as val loss = {val_loss:.4f}')


def train_byol(encoder, opt, device):
    train_loader, val_loader = get_loaders(opt)
    total_training_steps = compute_total_training_steps(train_loader, opt)

    learner = BYOL(
        encoder=encoder,
        feature_dim=opt.feature_dim,
        tau_base=opt.tau_base,
        total_training_steps=total_training_steps,
    )

    optimizer = optim.Adam(learner.trainable_parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    best_loss = np.inf
    t_epoch = tqdm(range(opt.epochs), desc='Epochs')
    for epoch in t_epoch:
        learner.train()
        t_batch = tqdm(train_loader, desc='Batches', leave=False)
        for i, (img_1, img_2, _) in enumerate(t_batch):
            current_training_steps = epoch * len(train_loader) + i

            img_1, img_2 = img_1.to(device), img_2.to(device)
            loss = learner(img_1, img_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / len(train_loader))
            learner.update_target_network(current_training_steps=current_training_steps)

            t_batch.set_postfix({'train loss': f'{loss.item():.4f}'})

        learner.eval()
        val_loss = []
        with torch.no_grad():
            for img, _, _ in val_loader:
                img = img.to(device)
                loss = learner(img, img)
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        t_epoch.set_postfix({'val loss': f'{val_loss:.4f}'})

        if val_loss < best_loss:
            best_loss = val_loss
            learner.save(dir_path=opt.output_dir)
            print(f'save model as val loss = {val_loss:.4f}')


def get_loaders(opt):
    train_transform, val_transform = get_transform(opt)

    train_dataset = ImagenetteDataset(Path(opt.img_dir) / 'train', opt.img_size, transform_1=train_transform, transform_2=train_transform)
    val_dataset = ImagenetteDataset(Path(opt.img_dir) / 'val', opt.img_size, transform_1=val_transform, transform_2=None)
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    return train_loader, val_loader
    

if __name__ == '__main__':
    main()
