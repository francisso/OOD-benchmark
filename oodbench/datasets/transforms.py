from typing import Sequence, Union

import numpy as np
from connectome import Transform, meta
from imops import zoom, label
from sklearn.model_selection import train_test_split

from amid.utils import propagate_none


Numeric = Union[float, int]


class Rescale(Transform):
    __inherit__ = True

    _new_spacing: Union[Sequence[Numeric], Numeric]
    _order: int = 1

    def _spacing(spacing, _new_spacing):
        _new_spacing = np.broadcast_to(_new_spacing, len(spacing)).copy()
        _new_spacing[np.isnan(_new_spacing)] = np.array(spacing)[np.isnan(_new_spacing)]
        return tuple(_new_spacing.tolist())

    def _scale_factor(spacing, _spacing):
        return np.float32(spacing) / np.float32(_spacing)

    def spacing(_spacing):
        return _spacing

    def image(image, _scale_factor, _order):
        return zoom(image.astype(np.float32), _scale_factor, order=_order)

    @propagate_none
    def mask(mask, _scale_factor, _order):
        return zoom(mask.astype(np.float32), _scale_factor, order=_order) > 0.5


def min_max_scale(image: np.ndarray):
    min_val = image.min()
    max_val = image.max()
    if min_val == max_val:
        raise ValueError('The scale range is zero.')
    return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=np.float32)


class ScaleIntensityMRI(Transform):
    __inherit__ = True

    _min_q: int = 1
    _max_q: int = 99

    def image(image, _min_q, _max_q):
        if _max_q <= _min_q:
            raise ValueError(f'`min_q` should be less than `max_q`; {_min_q} and {_max_q} are given.')

        image = np.float32(image)
        image = np.clip(image, *np.percentile(image, [_min_q, _max_q]))
        return min_max_scale(image)


class ScaleIntensityCT(Transform):
    __inherit__ = True

    # Standard lung window by default:
    _min_hu: float = -1350
    _max_hu: float = 300

    def image(image, _min_hu, _max_hu):
        if _max_hu <= _min_hu:
            raise ValueError(f'`min_hu` should be less than `max_hu`; {_min_hu} and {_max_hu} are given.')

        image = np.clip(image, _min_hu, _max_hu)
        return min_max_scale(image)


class AddShape(Transform):
    __inherit__ = True

    def shape(image):
        return image.shape


class TumorCenters(Transform):
    __inherit__ = True

    def _labels_n_labels(mask):
        return label(mask > 0.5, return_num=True, connectivity=3)

    def n_tumors(_labels_n_labels):
        return _labels_n_labels[1]

    def tumor_centers(_labels_n_labels):
        labels, n_labels = _labels_n_labels
        return np.int16([np.round(np.mean(np.argwhere(labels == i), axis=0)) for i in range(1, n_labels + 1)])


class CanonicalMRIOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.transpose(image, (1, 0, 2))[..., ::-1]

    def mask(mask):
        return np.transpose(mask, (1, 0, 2))[..., ::-1]

    def spacing(spacing):
        return tuple(np.array(spacing)[[1, 0, 2]].tolist())


class TrainTestSplit(Transform):
    __inherit__ = True

    _test_size: float = 1
    _random_state: Union[int, None] = None

    def _train_test_split(ids, _test_size, _random_state):
        if _test_size < 1:
            return train_test_split(ids, test_size=_test_size, random_state=_random_state)
        else:
            return [], ids

    @meta
    def train_ids(_train_test_split):
        return _train_test_split[0]

    @meta
    def test_ids(_train_test_split):
        return _train_test_split[1]
