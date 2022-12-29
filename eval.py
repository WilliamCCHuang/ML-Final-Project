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
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--evaluation-scheme', type=str, default='linear')
    parser.add_argument('--num-layers', type=int, default=50)
    parser.add_argument('--wide-scale', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)  # 4096 in paper
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='seed')

    opt = parser.parse_args()

    return opt


def check_opt(opt):
    opt.evaluation_scheme in ['linear']

    opt.num_layers in [50, 101, 152, 200]
    opt.wide_scale in [1, 2, 3, 4]

    if 'imagenette' in opt.img_dir:
        opt.num_classes = 10
    opt.last_fc = False

    if opt.output_dir is None:
        raise RuntimeError('The argument `output-dir` must be assign')
    opt.output_dir = Path(opt.output_dir)
    if not opt.output_dir.exists():
        opt.output_dir.mkdir(parents=True, exist_ok=True)

    opt.checkpoint_path = Path(opt.checkpoint_path)
    if not opt.checkpoint_path.exists():
        raise FileNotFoundError()

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

    encoder = load_encoder(opt, device)
    opt.feature_dim = get_feature_dim(encoder, opt.img_size, device)

    linear_eval(encoder, opt, device)


def load_encoder(opt, device):
    cpu_device = torch.device('cpu')
    encoder = get_resnet(opt).to(cpu_device)

    checkpoint = torch.load(str(opt.checkpoint_path))
    if 'encoder' in checkpoint:
        encoder_state_dict = checkpoint['encoder']
    elif 'online_encoder' in checkpoint:
        encoder_state_dict = checkpoint['encoder']

    encoder.load_state_dict(encoder_state_dict).to(device)

    return encoder


def linear_eval(encoder, opt, device):
    train_loader, val_loader = get_loaders(opt)
    transform = get_transform(opt)
    train_loader.dataset.transform = transform
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    classifier = nn.Linear(opt.feature_dim, opt.num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr)

    best_acc = 0
    t_epoch = tqdm(range(opt.epochs), desc='Epochs')
    for _ in t_epoch:
        t_batch = tqdm(train_loader, desc='Batches')
        for img, label in t_batch:
            bz = img.shape[0]

            img = img.to(device)
            label = label.to(device)

            with torch.no_grad():
                z = encoder(img)
                z = z.view(bz, -1)

            y_prob = classifier(z)
            loss = loss_fn(y_prob, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_batch.set_postfix({'train loss': f'{loss.item():.4f}'})

        with torch.no_grad():
            val_acc = 0
            val_loss = []
            for img, label in val_loader:
                bz = img.shape[0]

                img = img.to(device)
                label = label.to(device)

                z = encoder(img)
                z = z.view(bz, -1)
                y_prob = classifier(z)
                loss = loss_fn(y_prob, label)
                val_loss.append(loss.item())

                _, y_pred = torch.max(y_prob, dim=1)
                val_acc += (label == y_pred).sum()

        val_loss = np.mean(val_loss)
        val_acc = val_acc.item() / len(val_loader.dataset)
        t_epoch.set_postfix({'val loss': f'{val_loss:.4f}', 'val acc': f'{val_acc:.4f}'})

        if val_acc > best_acc:
            best_acc = val_acc

            checkpoint = {
                'feature_dim': opt.feature_dim,
                'num_classes': opt.num_classes,
                'classifier': classifier.state_dict()
            }
            torch.save(checkpoint, str(opt.output_dir / 'linear.pt'))
            print(f'save model as val acc = {val_acc:.4f}')


def get_loaders(opt):
    train_dataset = ImagenetteDataset(Path(opt.img_dir) / 'train', opt.img_size)
    val_dataset = ImagenetteDataset(Path(opt.img_dir) / 'val', opt.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    return train_loader, val_loader
    

if __name__ == '__main__':
    main()
