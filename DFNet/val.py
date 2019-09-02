import numpy as np
import math
import torch
from tqdm import tqdm_notebook
from torch.utils.data import DataLoader


def array_flatten(x):
    return x.reshape((x.shape[0], -1))


def psnr_vectorized(x, y):
    mse = (x - y) ** 2
    mse = array_flatten(mse).mean(axis=1)
    mse = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse


def l1_loss_vectorized(x, y):
    mae = array_flatten(np.abs(x - y) / 256.).sum(axis=1) / 3
    return mae


def psnr_tensor(input, target):
    mse = torch.flatten(
        torch.nn.functional.mse_loss(input, target, reduction='none'), start_dim=1).mean(dim=1)
    psnr_v = -20 * torch.log(torch.sqrt(mse))
    return psnr_v.mean()


def l1_loss_tensor(input, target):
    l1 = torch.flatten((input - target).abs(), start_dim=1).sum(dim=1) * (255 / 256)
    return l1.mean()


def psnr(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse == 0:
        return math.inf
    return 20 * math.log10(255.0 / math.sqrt(mse))


def l1_loss(x, y):
    return np.sum(np.abs(x / 256. - y / 256.)) / 3.


def validation(orig_dataset, inpainted, metrics={'PSNR': psnr, 'L1 loss': l1_loss}, tqdm_on=True):
    val_loader = DataLoader(orig_dataset, batch_size=1, shuffle=False)
    results = {k: 0 for k in metrics.keys()}
    
    gen = enumerate(val_loader)
    if tqdm_on:
        gen = tqdm_notebook(gen)
    
    for i, (img, _) in gen:
        img = np.array(img)[0]
        for k, func in metrics.items():
            results[k] += func(img, inpainted[i])
    for k in results:
        results[k] /= len(orig_dataset)
    return results
