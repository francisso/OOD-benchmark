import numpy as np
from amid.internals import CacheToDisk, CacheColumns
from amid.medseg9 import Medseg9
from connectome import Transform, Apply, chained

from ...const import CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TrainTestSplit


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


Medseg9 = chained(
    TrainTestSplit(),
    RenameFieldsMedseg9(),
    CanonicalOrientation(),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(Medseg9)
