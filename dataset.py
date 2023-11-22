import os
from pathlib import Path

import scipy
import torch
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from PIL.Image import Image
from torch.utils.data import Dataset

import config


class StanfordCarsDataset(Dataset):
    def __init__(self, source_path: str, annotation_path: str, transforms: callable = None):
        self.source_paths = self.__load_imgs_paths(source_path)
        self.labels = self.__load_labels(annotation_path)
        self.transforms = transforms

    def __load_imgs_paths(self, path: Path) -> list[Path]:
        if not os.path.exists(path):
            raise ValueError("Not found file %s" % str(path))
        return [path / filename for filename in os.listdir(path) if os.path.isfile(path / filename)]

    def __load_labels(self, path: Path):
        if not os.path.exists(path):
            raise ValueError("Not found file %s" % str(path))

        lable_mat = scipy.io.loadmat(path)
        labels = {}
        for arr in lable_mat['annotations'][0]:
            filename, label = str(arr[5][0]), int(arr[4][0, 0]) - 1
            labels[filename] = label

        return labels

    @property
    def features_amount(self) -> int:
        return len(set(self.labels))

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        img_path = self.source_paths[index]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[os.path.basename(img_path)]

        if self.transforms:
            img = self.transforms(image=np.array(img))['image']
            img = img.to(torch.float)

        return img, label


transform = A.Compose([
    A.Resize(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(width=224, height=224),
    ToTensorV2(),
])

