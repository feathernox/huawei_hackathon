import os
from PIL import Image
import numpy as np
import json
import cv2
import re
import random
from skimage.morphology import binary_erosion, square
from torch.utils.data import Dataset, DataLoader
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb


def mask2img(mask, height, width):
    img = np.ones((height, width)).astype("uint8") * 255
    for m in mask:
        cv2.rectangle(img, (m[1], m[0]), (m[3], m[2]), 0, thickness=-1)
    return img


class FinalDataset(Dataset):
    def __init__(self, path, mask_path, transform=None, expand_mask=0):
        self.path = path
        self.mask_path = mask_path
        self.transform = transform
        self.expand_mask = expand_mask

        self.indices = [im.split('.')[0] for im in os.listdir(mask_path)]
        self.indices = sorted(self.indices, key=lambda x: int(x))
        #
        # self.imgs = []
        # self.masks = []
        # for m in masks:
        #     m = m.split()
        #     self.imgs.append(m[0])
        #     self.masks.append(json.loads(' '.join(m[1:])))
        # self.masks = [np.array(m) for m in self.masks]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img_idx = self.indices[i]

        img_masked = np.asarray(Image.open(os.path.join(self.path, img_idx + '.png')))
        mask = np.asarray(Image.open(os.path.join(self.mask_path, img_idx + '.png')))
        if mask.shape == 3:
            mask = mask.mean(axis=1)

        mask = (~(mask > 0)).astype('uint8') * 255

        if self.expand_mask > 0:
            mask = binary_erosion(mask, square(self.expand_mask)).astype('uint8') * 255

        if self.transform is not None:
            transformed = self.transform(image=img_masked, mask=mask)
            img_masked, mask = transformed['image'], transformed['mask']
        return img_masked, mask


class InpaintingDataset(Dataset):
    def __init__(self, path, path_txt, transform=None, train=True, expand_mask=0):
        self.train = train
        self.path = path
        self.path_txt = path_txt
        self.transform = transform
        self.expand_mask = expand_mask
        with open(self.path_txt, 'r') as f:
            masks = f.readlines()

        self.imgs = []
        self.masks = []
        for m in masks:
            m = m.split()
            self.imgs.append(m[0])
            self.masks.append(json.loads(' '.join(m[1:])))
        self.masks = [np.array(m) for m in self.masks]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img_idx = self.imgs[i]
        if self.train:
            img_masked = np.asarray(Image.open(os.path.join(self.path, img_idx + '_gt.png')))
        else:
            img_masked = np.asarray(Image.open(os.path.join(self.path, img_idx + '.png')))

        height, width, _ = img_masked.shape

        mask = np.ones((height, width)).astype("uint8") * 255
        for m in self.masks[i]:
            cv2.rectangle(mask, (m[1], m[0]), (m[3], m[2]), 0, thickness=-1)

        if self.expand_mask > 0:
            mask = binary_erosion(mask, square(self.expand_mask)).astype('uint8') * 255

        if self.transform is not None:
            transformed = self.transform(image=img_masked, mask=mask)
            img_masked, mask = transformed['image'], transformed['mask']
        return img_masked, mask


class InpaintingPoutyneDataset(InpaintingDataset):
    def __init__(self, path, path_txt, transform=None, expand_mask=0):
        super().__init__(path, path_txt, train=False, transform=None, expand_mask=expand_mask)
        self.transform_poutyne = transform

    def __getitem__(self, i):
        img_masked, mask = super().__getitem__(i)

        img_idx = self.imgs[i]
        img_orig = np.asarray(Image.open(os.path.join(self.path, img_idx + '_gt.png')))

        if self.transform_poutyne is not None:
            transformed = self.transform_poutyne(image=img_masked, mask=mask, orig=img_orig)
            img_masked, mask, img_orig = transformed['image'], transformed['mask'], transformed['orig']

        return (img_masked, mask), img_orig


class MaskGetter():
    def __init__(self, masks, min_num=1, max_num=4,
                 width=500, height=600, w_shift=0, h_shift=0):
        self.masks = masks
        self.min_num = min_num
        self.max_num = max_num
        self.width = width
        self.height = height
        self.w_shift = w_shift
        self.h_shift = h_shift

    def __call__(self):
        n = random.randint(self.min_num, self.max_num)
        mask = random.sample(self.masks, n)
        mask = [self._apply_flips(m) for m in mask]
        return mask

    def _apply_flips(self, mask):
        w_shift = random.randint(0, self.w_shift)
        mask[0], mask[2] = mask[0] + w_shift, mask[2] + w_shift

        h_shift = random.randint(0, self.h_shift)
        mask[1], mask[3] = mask[1] + h_shift, mask[3] + h_shift

        state = random.randint(0, 3)
        if state % 2:
            mask = [self.width - mask[2], mask[1], self.width - mask[0], mask[3]]
        if state / 2:
            mask = [mask[0], self.height - mask[3], mask[2], self.height - mask[1]]
        return mask


class InpaintingEdgeConnectDataset(InpaintingDataset):
    def __init__(self, path, path_txt, transform=None, expand_mask=0, sigma=2):
        super().__init__(path, path_txt, train=False, transform=None, expand_mask=expand_mask)
        self.transform_edgeconnect = transform
        self.sigma = sigma

    def make_edge(self, img, mask):
        mask = (1 - mask / 255).astype(np.bool)
        return canny(img, sigma=self.sigma, mask=mask).astype(np.float)

    def __getitem__(self, i):
        img_masked, mask = super().__getitem__(i)
        mask = ~(mask > 0)
        mask = (mask * 255).astype('uint8')

        img_gray = rgb2gray(img_masked)
        edge = self.make_edge(img_gray, mask)

        img_gray, edge = img_gray[..., None] * 255, edge[..., None] * 255

        if self.transform_edgeconnect is not None:
            transformed = self.transform_edgeconnect(image=img_masked, mask=mask, gray=img_gray, edge=edge)
            img_masked, img_gray, edge, mask = (transformed[attr] for attr in ['image', 'gray', 'edge', 'mask'])

        return img_masked, img_gray, edge, mask
