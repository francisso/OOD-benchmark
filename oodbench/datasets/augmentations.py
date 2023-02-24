import os
from collections import defaultdict
from typing import Union
from functools import lru_cache
import hashlib

import numpy as np
from tqdm import tqdm
from connectome import Transform

from dpipe.io import save_json, load_pred, save, load, load_json



def elastic_transform(img: np.ndarray, mask: np.ndarray = None, param: float = 0.5, random_state: int = 5):
    from albumentations.augmentations.geometric.transforms import ElasticTransform
    import cv2
    from skimage.measure import label

    fill_value = int(img.min()) if img is not None else 0
    shape = img.shape[:2] if img is not None else mask.shape[:2]
    border_mode = cv2.BORDER_CONSTANT

    t = ElasticTransform(alpha=12000 * param, sigma=np.mean(shape) / 11., alpha_affine=0,
                         always_apply=True, border_mode=border_mode, value=fill_value)
    t.set_deterministic(True)

    if mask is not None:
        mask_new = t.apply_to_mask(mask, random_state=random_state)

    if img is not None:
        img_new = t.apply(img, random_state=random_state)
        lbl = label(np.pad(img_new, [[1, 1], [1, 1], [0, 0]]) == 0, connectivity=3)
        lbl = lbl[1:-1, 1:-1, :]

        # always lbl == 1 because algorithm starts at the corner of the image.
        img_new[lbl == 1] = fill_value

    if mask is None:
        return img_new
    if img is None:
        return mask_new
    return img_new, mask_new


def blur_transform(img: np.ndarray, param: float = 0.5, random_state: int = 5):
    from skimage.filters import gaussian
    return gaussian(np.float32(img), sigma=5 * param, preserve_range=True)


def slice_drop_transform(img: np.ndarray, param: float = 0.5,
                         random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    drop_indexes = np.nonzero(random_state.binomial(1, param, size=img.shape[-1]))[0]

    fill_value = img.min()

    img_new = img.copy()
    for i in drop_indexes:
        img_new[..., i] = np.zeros_like(img_new[..., i]) + fill_value

    return img_new


def sample_box(img: np.ndarray, param: float = 0.5, random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    img_shape = np.int16(img.shape)
    box_shape = np.int16(np.round(np.float32(img.shape) * param))
    center_min = box_shape // 2
    center_max = img_shape - box_shape // 2

    center = np.int16([random_state.randint(cmin, cmax) for cmin, cmax in zip(center_min, center_max)])
    return [center - box_shape // 2, center + box_shape // 2]


def min_max_scale(img: np.ndarray):
    img = np.float32(img)
    img -= img.min()
    img /= img.max()
    return img


def min_max_descale(img: np.ndarray, minv, maxv):
    img = np.float32(img)
    img *= (maxv - minv)
    img += minv
    return img


def contrast_transform(img: np.ndarray, param: float = 0.5,
                       random_state: int = 5):
    from skimage.exposure import adjust_gamma

    random_state_np = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    minv, maxv = crop.min(), crop.max()

    while minv == maxv:
        print('resampling bbox because crop.min() == crop.max()', flush=True)
        random_state += 1
        random_state_np = np.random.RandomState(random_state)

        box = sample_box(img, param, random_state)
        crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
        minv, maxv = crop.min(), crop.max()

    gamma = 4 if (random_state_np.random_sample() >= 0.5) else 0.25
    crop_corrected = adjust_gamma(min_max_scale(crop), gamma=gamma)
    crop_corrected = min_max_descale(crop_corrected, minv, maxv)

    img_new = np.copy(img)
    img_new[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]] = crop_corrected

    return img_new


def corruption_transform(img: np.ndarray, param: float = 0.5,
                         random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    minv, maxv = crop.min(), crop.max()

    crop_corrupted = min_max_descale(random_state.rand(*crop.shape), minv, maxv)

    img_new = np.copy(img)
    img_new[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]] = crop_corrupted

    return img_new


def pixel_shuffling_transform(img: np.ndarray, param: float = 0.5,
                              random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    crop_shape = np.copy(np.asarray(crop.shape))

    crop_shuffled = crop.ravel()
    random_state.shuffle(crop_shuffled)
    crop_shuffled = np.reshape(crop_shuffled, crop_shape)

    img_new = np.copy(img)
    img_new[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]] = crop_shuffled

    return img_new

aug_list = {
        'elastic.transform': elastic_transform, 
        'blur.transform': blur_transform, 
        'slicedrop.transform': slice_drop_transform, 
        'contrast.transform': contrast_transform,
        'corruption.transform': corruption_transform,
        'pixelshuffling.transform': pixel_shuffling_transform,
    }