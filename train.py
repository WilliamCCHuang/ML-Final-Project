import argparse
from tqdm import tqdm

import torch.optim as optim
from torch.utils.data import Dataloader

from byol import BYOL
from datasets import ImageNetDataset
from resnet import get_resnet
from utils import seed_everything, get_feature_dim, get_augment_funcs, compute_total_training_steps


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--training-scheme', type=str, default='byol')
    parser.add_argument('--num-layers', type=int, default=50)
    parser.add_argument('--wide-scale', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256) # 4096 in paper
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

    # TODO: for other datasets
    if opt.dataset == 'imagenet':
        opt.num_classes = 1000

    opt.lr = opt.lr_base * opt.batch_size / 256


def main():
    opt = parse_opt()
    check_opt(opt)
    seed_everything(opt.seed)

    encoder = get_resnet(opt)
    opt.feature_dim = get_feature_dim(encoder, opt.img_size)


    scheme_func = None
    if opt.training_scheme == 'supervised':
        scheme_func = train_supervised
    elif opt.training_scheme == 'simclr':
        scheme_func = train_simclr
    else:
        scheme_func = train_byol

    opt.augment_funcs = get_augment_funcs(opt)
    scheme_func(encoder, opt)


def train_supervised(encoder, opt):
    raise NotImplementedError  # TODO:


def train_simclr(loader, encoder, opt):
    raise NotImplementedError  # TODO:


def train_byol(encoder, opt):
    loader = get_loaders(opt)
    augment_func1, augment_func2 = get_augment_funcs(opt)
    total_training_steps = compute_total_training_steps(loader, opt)

    learner = BYOL(
        encoder=encoder,
        feature_dim=opt.feature_dim,
        augment_func1=augment_func1,
        augment_func2=augment_func2,
        tau_base=opt.tau_base,
        total_training_steps=total_training_steps,
    )

    optimizer = optim.Adam(learner.parameters(), lr=opt.lr)

    for epoch in tqdm(range(opt.epochs), desc='Epochs', leave=False):
        for i, (img, _) in enumerate(tqdm(loader)):
            loss = learner(img)
            optimizer.zero_grad()
            loss.backward()
            learner.update_target_network(current_training_steps=epoch * len(loader) + i)


def get_loaders(opt):
    dataset = ImageNetDataset()
    # TODO: split train, val

    

if __name__ == '__main__':
    main()
