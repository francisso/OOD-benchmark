from typing import Union

import numpy as np
from amid.internals import CacheToDisk, CacheColumns
from amid.lits import LiTS as LiTS_AMID, CanonicalCTOrientation, Rescale
from connectome import Apply, Transform, chained

from ..transforms import AddShape, Identity, ScaleIntensityCT, TrainTestSplit
from ..wrappers import Proxy
from ...const import CT_COMMON_SPACING
from ...typing import PathLike


__all__ = ['LiTS', ]


class ChangeFieldsLiTS(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


class LiTS(Proxy):
    def __init__(self, root: Union[PathLike, None] = None, use_caching: bool = True):
        dataset_chained = chained(
            TrainTestSplit(),
            CanonicalCTOrientation(),
            Rescale(new_spacing=CT_COMMON_SPACING),
            ChangeFieldsLiTS(),
            ScaleIntensityCT(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids',)) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing',)),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(LiTS_AMID)

        super().__init__(dataset_chained(root))
