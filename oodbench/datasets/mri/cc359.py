from typing import Union

import numpy as np
from amid.cc359 import CC359 as CC359_AMID, CanonicalMRIOrientation, Rescale
from amid.internals import CacheColumns, CacheToDisk
from connectome import Apply, Transform, chained

from ..transforms import AddShape, Identity, ScaleIntensityMRI, TrainTestSplit
from ..wrappers import Proxy
from ...const import MRI_COMMON_SPACING
from ...typing import PathLike


__all__ = ['CC359', ]


class RenameFieldsCC359(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


class CC359(Proxy):
    def __init__(self, root: Union[PathLike, None] = None, use_caching: bool = True):
        dataset_chained = chained(
            TrainTestSplit(),
            CanonicalMRIOrientation(),
            Rescale(new_spacing=MRI_COMMON_SPACING),
            RenameFieldsCC359(),
            ScaleIntensityMRI(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids', )) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing', )) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(CC359_AMID)

        super().__init__(dataset_chained(root))
