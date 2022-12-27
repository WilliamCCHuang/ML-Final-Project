from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ImagenetteDataset(Dataset):

    def __init__(self, dir_path):
        super().__init__()

        self.dir_path = Path(dir_path)

        if not self.dir_path.exists():
            raise FileNotFoundError(f'Can not find {str(self.dir_path)}')

        self.class2idx = {str(class_name): i for i, class_name in enumerate(self.dir_path.iterdir())}

        self.img_infos = []
        for class_name, class_idx in self.class2idx.items():
            if not class_name.startswith('n'):
                continue
            
            class_dir = self.dir_path / class_name

            self.img_infos += [(img_path, class_idx)for img_path in class_dir.iterdir()]

        print('Imagenette Info:')
        print(f'  # images = {len(self.img_infos)}')
        print(f'  # classes = {len(self.class2idx)}')

    def __len__(self):
        raise len(self.img_infos)

    def __getitem__(self, idx):
        img_path, class_idx = self.img_infos[idx]

        img = torch.from_numpy(Image.open(str(img_path))).transpose(2, 0, 1)

        return img, class_idx
