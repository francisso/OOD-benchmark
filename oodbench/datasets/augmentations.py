from typing import Union

import cv2
import numpy as np
import torchio as tio
from albumentations.augmentations.geometric.transforms import ElasticTransform
from imops import label
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian


__all__ = ['decode_id', 'AUGM_LIST', ]


def decode_id(i):
    base_id, augm_name, scale = i.split(":")
    scale = float(scale)
    return base_id, augm_name, scale


def elastic_transform(img: np.ndarray, mask: np.ndarray = None, param: float = 0.5, random_state: int = 5):

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
    return gaussian(np.float32(img), sigma=5 * param, preserve_range=True)


def slice_drop_transform(img: np.ndarray, param: float = 0.5,
                         random_state: Union[int, np.random.RandomState] = 5):
    # FIXME: remove `np.random.RandomState` option?
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    drop_indexes = np.nonzero(random_state.binomial(
        1, param, size=img.shape[-1]))[0]

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


def contrast_transform(img: np.ndarray, param: float = 0.5, random_state: int = 5):
    random_state_np = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    minv, maxv = crop.min(), crop.max()

    while minv == maxv:
        print('resampling bbox because crop.min() == crop.max()', flush=True)  # TODO: warn?
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
    # FIXME: remove `np.random.RandomState` option?
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    minv, maxv = crop.min(), crop.max()

    crop_corrupted = min_max_descale(
        random_state.rand(*crop.shape), minv, maxv)

    img_new = np.copy(img)
    img_new[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]] = crop_corrupted

    return img_new


def pixel_shuffling_transform(img: np.ndarray, param: float = 0.5,
                              random_state: Union[int, np.random.RandomState] = 5):
    # FIXME: remove `np.random.RandomState` option?
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


def ghosting_transform(image, param: int = 3, random_state: int = 5):
    param = int(param)  # TODO: weak behavior?
    assert 1 <= param <= 5  # FIXME: raise ValueError
    num_ghosts = [(1, 2), (2, 4), (4, 6), (6, 7), (7, 10)]
    intensity = [(.1, .1), (.1, .2), (.3, .4), (.4, .6), (.6, .99)]
    transform = tio.Compose({
        tio.RandomGhosting(num_ghosts=num_ghosts[param-1], intensity=intensity[param-1]): 1.0,
    })
    si = tio.ScalarImage(tensor=image[None, ...])
    with transform._use_seed(random_state):
        si = transform.apply_transform(si)
    return si.tensor[0].numpy()


def anisotropy_transform(image, param: int = 3, random_state: int = 5):
    param = int(param)  # TODO: weak behavior?
    assert 1 <= param <= 5  # FIXME: raise ValueError
    downsmapling = [(1.1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 6)]
    transform = tio.Compose({
        tio.RandomAnisotropy(downsampling=downsmapling[param-1]): 1.0,
    })
    si = tio.ScalarImage(tensor=image[None, ...])
    with transform._use_seed(random_state):
        si = transform.apply_transform(si)
    return si.tensor[0].numpy()


def spike_transform(image, param: int = 3, random_state: int = 5):
    param = int(param)  # TODO: weak behavior?
    assert 1 <= param <= 5  # FIXME: raise ValueError
    intensity = [(0, 0.250), (.25, .35), (.35, .5), (.5, .7), (.7, 1.2)]
    transform = tio.Compose({
        tio.RandomSpike(intensity=intensity[param-1]): 1.0,
    })
    si = tio.ScalarImage(tensor=image[None, ...])
    with transform._use_seed(random_state):
        si = transform.apply_transform(si)
    return si.tensor[0].numpy()


def bfield_transform(image, param: int = 3, random_state: int = 5):
    # this one is slow
    param = int(param)  # TODO: weak behavior?
    assert 1 <= param <= 5  # FIXME: raise ValueError
    coefficients = [(0.01, 0.1), (.1, .2), (.2, .3), (.4, .5), (.5, .6)]
    transform = tio.Compose({
        tio.RandomBiasField(coefficients=coefficients[param-1]): 1.0,
    })
    si = tio.ScalarImage(tensor=image[None, ...])
    with transform._use_seed(random_state):
        si = transform.apply_transform(si)
    return si.tensor[0].numpy()


def motion_transform(image, param: int = 3, random_state: int = 5):
    param = int(param)  # TODO: weak behavior?
    assert 1 <= param <= 5  # FIXME: raise ValueError
    num_transforms = [1, 1, 2, 2, 3]
    scale = [(.01, .12), (.12, .15), (.15, .2), (.2, .3), (.3, .5)]

    transform = tio.Compose({
        tio.RandomMotion(degrees=scale[param-1], translation=scale[param-1],
                         num_transforms=num_transforms[param-1]): 1.0,
    })
    si = tio.ScalarImage(tensor=image[None, ...])
    with transform._use_seed(random_state):
        si = transform.apply_transform(si)
    return si.tensor[0].numpy()


AUGM_LIST = {
    'elastic.transform': elastic_transform,
    'blur.transform': blur_transform,
    'slicedrop.transform': slice_drop_transform,
    'contrast.transform': contrast_transform,
    'corruption.transform': corruption_transform,
    'pixelshuffling.transform': pixel_shuffling_transform,
    'ghosting.transform': ghosting_transform,
    'anisotropy.transform': anisotropy_transform,
    'spike.transform': spike_transform,
    'bfield.transform': bfield_transform,
    'motion.transform': motion_transform,
}
