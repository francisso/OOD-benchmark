import numpy as np
from amid.internals import CacheColumns, CacheToDisk
from amid.medseg9 import Medseg9 as Medseg9_AMID
from connectome import Apply, Transform, chained

from ..wrappers import Proxy
from ..transforms import AddShape, Rescale, ScaleIntensityCT, TrainTestSplit, Identity
from ...config import PATH_MEDSEG9_RAW, USE_CACHING
from ...const import CT_COMMON_SPACING
from ...typing import OptPathLike


__all__ = ['Medseg9', ]


class RenameFieldsMedseg9(Transform):
    __exclude__ = ('affine', 'voxel_spacing', )

    def mask(image):
        return np.zeros_like(image, dtype=bool)


class CanonicalOrientation(Transform):
    __inherit__ = True
    _flip_x: bool = True

    def image(image, _flip_x):
        return np.transpose(image, (1, 0, 2))

    def mask(mask, _flip_x):
        return np.transpose(mask, (1, 0, 2))

    def spacing(spacing):
        return tuple(np.array(spacing)[[1, 0, 2]].tolist())


class Medseg9(Proxy):
    def __init__(self, root: OptPathLike = PATH_MEDSEG9_RAW, use_caching: bool = USE_CACHING):
        dataset_chained = chained(
            TrainTestSplit(),
            RenameFieldsMedseg9(),
            CanonicalOrientation(),
            Rescale(new_spacing=CT_COMMON_SPACING),
            ScaleIntensityCT(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids', )) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing', )) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(Medseg9_AMID)

        super().__init__(dataset_chained(root))
