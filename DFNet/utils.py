from pathlib import Path

import cv2
import math
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from albumentations.core.transforms_interface import BasicTransform, DualTransform, to_tuple
from albumentations.augmentations import functional


def img_to_tensor(im):
    img = np.ascontiguousarray(im.transpose(2, 0, 1)).astype(np.uint8)
    img = torch.from_numpy(img)
    img = img.float().div(255)
    return img


def mask_to_tensor(mask):
    mask = np.ascontiguousarray(np.expand_dims(mask, 0)).astype(np.uint8)
    mask = torch.from_numpy(mask)
    mask = mask.float().div(255)
    return mask


class ToTensor(BasicTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True, p=1.)

    def __call__(self, **kwargs):
        kwargs.update({'image': img_to_tensor(kwargs['image'])})
        kwargs.update({'mask': mask_to_tensor(kwargs['mask'])})

        for k, v in kwargs.items():
            if self._additional_targets.get(k) == 'image':
                kwargs.update({k: img_to_tensor(kwargs[k])})
            if self._additional_targets.get(k) == 'mask':
                kwargs.update({k: mask_to_tensor(kwargs[k])})
        return kwargs

    @property
    def targets(self):
        raise NotImplementedError


class FixedRotate(DualTransform):
    def __init__(self, angle=0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101,
                 value=None, mask_value=None):
        super(FixedRotate, self).__init__(always_apply=True, p=1)
        self.angle = angle
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return functional.rotate(img, self.angle, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, **params):
        return functional.rotate(img, self.angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def apply_to_bbox(self, bbox, **params):
        return functional.bbox_rotate(bbox, self.angle, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return functional.keypoint_rotate(keypoint, self.angle, **params)

    def get_transform_init_args_names(self):
        return ('angle', 'interpolation', 'border_mode', 'value', 'mask_value')


def to_numpy(tensor):
    tensor = tensor.mul(255).byte().data.cpu().numpy()
    tensor = np.transpose(tensor, [0, 2, 3, 1])
    return tensor


def resize_like(x, target, mode='bilinear'):
    return F.interpolate(x, target.shape[-2:], mode=mode, align_corners=False)


def list2nparray(lst, dtype=None):
    """fast conversion from nested list to ndarray by pre-allocating space"""
    if isinstance(lst, np.ndarray):
        return lst
    assert isinstance(lst, (list, tuple)), 'bad type: {}'.format(type(lst))
    assert lst, 'attempt to convert empty list to np array'
    if isinstance(lst[0], np.ndarray):
        dim1 = lst[0].shape
        assert all(i.shape == dim1 for i in lst)
        if dtype is None:
            dtype = lst[0].dtype
            assert all(i.dtype == dtype for i in lst), \
                'bad dtype: {} {}'.format(dtype, set(i.dtype for i in lst))
    elif isinstance(lst[0], (int, float, complex, np.number)):
        return np.array(lst, dtype=dtype)
    else:
        dim1 = list2nparray(lst[0])
        if dtype is None:
            dtype = dim1.dtype
        dim1 = dim1.shape
    shape = [len(lst)] + list(dim1)
    rst = np.empty(shape, dtype=dtype)
    for idx, i in enumerate(lst):
        rst[idx] = i
    return rst


def get_img_list(path):
    return sorted(list(Path(path).glob('*.png'))) + \
           sorted(list(Path(path).glob('*.jpg'))) + \
           sorted(list(Path(path).glob('*.jpeg')))


def gen_miss(img, mask, output):
    imgs = get_img_list(img)
    masks = get_img_list(mask)
    print('Total images:', len(imgs), len(masks))

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    for i, (img, mask) in tqdm(enumerate(zip(imgs, masks))):
        path = out.joinpath('miss_%04d.png' % (i + 1))
        img = cv2.imread(str(img), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img.shape[:2][::-1])
        mask = mask[..., np.newaxis]
        miss = img * (mask > 127) + 255 * (mask <= 127)
        cv2.imwrite(str(path), miss)


def merge_imgs(dirs, output, row=1, gap=2, res=512):
    image_list = [get_img_list(path) for path in dirs]
    img_count = [len(image) for image in image_list]
    print('Total images:', img_count)
    assert min(img_count) > 0, 'Please check the path of empty folder.'

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_img = len(dirs)
    row = row
    column = (n_img - 1) // row + 1
    print('Row:', row)
    print('Column:', column)

    for i, unit in tqdm(enumerate(zip(*image_list))):
        name = output_dir.joinpath('merge_%04d.png' % i)
        merge = np.ones([
            res * row + (row + 1) * gap, res * column + (column + 1) * gap, 3], np.uint8) * 255
        for j, img in enumerate(unit):
            r = j // column
            c = j - r * column
            img = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if img.shape[:2] != (res, res):
                img = cv2.resize(img, (res, res))
            start_h, start_w = (r + 1) * gap + r * res, (c + 1) * gap + c * res
            merge[start_h: start_h + res, start_w: start_w + res] = img
        cv2.imwrite(str(name), merge)
