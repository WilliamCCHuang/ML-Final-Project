from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor


class ImagenetteDataset(Dataset):

    def __init__(self, dir_path, img_size, transform_1, transform_2):
        super().__init__()

        self.dir_path = Path(dir_path)
        self.img_size = img_size, img_size
        self.transform_1 = transform_1
        self.transform_2 = transform_2

        if not self.dir_path.exists():
            raise FileNotFoundError(f'Can not find {str(self.dir_path)}')

        self.class2idx = {}
        idx = 0
        for class_path in self.dir_path.iterdir():
            class_name = str(class_path).split('/')[-1]

            if not class_name.startswith('n'):
                continue

            self.class2idx[class_name] = idx
            idx += 1

        self.img_infos = []
        for class_name, class_idx in self.class2idx.items():
            class_dir = self.dir_path / class_name

            self.img_infos += [(img_path, class_idx) for img_path in class_dir.iterdir()]

        print('Imagenette Info:')
        print(f'  # images = {len(self.img_infos)}')
        print(f'  # classes = {len(self.class2idx)}')

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        img_path, class_idx = self.img_infos[idx]
        img = Image.open(str(img_path)).resize((160, 160))
        img = PILToTensor()(img).float()  # (1 or 3, 160, 160)

        if img.shape[0] == 1:
            img = torch.cat((img, img, img), dim=0)

        img_1 = self.transform_1(img)

        img_2 = 0
        if self.transform_2 is not None:
            img_2 = self.transform_2(img)

        return img_1, img_2, class_idx
