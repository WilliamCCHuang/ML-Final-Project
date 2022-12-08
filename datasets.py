from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class ImageNetDataset(Dataset):

    def __init__(self, dir_path):
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
