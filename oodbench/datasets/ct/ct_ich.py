from typing import Union

import numpy as np
from amid.ct_ich import CT_ICH as CT_ICH_AMID
from amid.internals import CacheToDisk, CacheColumns
from connectome import Transform, Apply, chained

from ..wrappers import Proxy
from ..transforms import AddShape, CanonicalOrientation, Identity, Rescale, ScaleIntensityCT, TrainTestSplit
from ...const import CT_COMMON_SPACING
from ...typing import PathLike


__all__ = ['CT_ICH', ]


class RenameFieldsCTICH(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


class CT_ICH(Proxy):
    def __init__(self, root: Union[PathLike, None] = None, use_caching: bool = True):
        dataset_chained = chained(
            TrainTestSplit(),
            RenameFieldsCTICH(),
            CanonicalOrientation(),
            Rescale(new_spacing=CT_COMMON_SPACING),
            ScaleIntensityCT(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids',)) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing',)) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(CT_ICH_AMID)

        super().__init__(dataset_chained(root))
