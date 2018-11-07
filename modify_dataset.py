# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetWithID(data.Dataset):

    def __init__(self, image_lists, transform=None):
        self.imgs = image_lists
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        return img1, img2

    def __len__(self):
        return len(self.imgs)


def create_dataset(image_lists):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # t = transforms.Compose([transforms.RandomResizedCrop(224),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         normalize])
    t = transforms.Compose([transforms.Resize(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])
    return DatasetWithID(image_lists, transform=t)
