from typing import Union

import numpy as np
from amid.internals import CacheColumns, CacheToDisk
from amid.midrc import MIDRC as MIDRC_AMID
from connectome import Apply, Filter, Transform, chained

from ..wrappers import Proxy
from ..transforms import AddShape, CanonicalOrientation, Identity, Rescale, ScaleIntensityCT, TrainTestSplit
from ...const import CT_COMMON_SPACING
from ...typing import PathLike


__all__ = ['MIDRC', ]


class ChangeFieldsMIDRC(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


class MIDRC(Proxy):
    def __init__(self, root: Union[PathLike, None] = None, use_caching: bool = True):
        dataset_chained = chained(
            Filter(lambda mask: (mask is not None) and (np.any(mask[0]) or np.any(mask[-1]))),
            TrainTestSplit(),
            ChangeFieldsMIDRC(),
            CanonicalOrientation(flip_x=False),
            Rescale(new_spacing=CT_COMMON_SPACING),
            ScaleIntensityCT(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids', )) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing', )) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(MIDRC_AMID)

        super().__init__(dataset_chained(root))
