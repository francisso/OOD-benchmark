import numpy as np
from amid import CacheToDisk
from amid.ct_ich import CT_ICH
from connectome import Transform, Apply, chained, CacheColumns

from ...const import CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TrainTestSplit


__all__ = ['CT_ICH', ]


class RenameFieldsCTICH(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)

    def spacing(voxel_spacing):
        return voxel_spacing


CT_ICH = chained(
    TrainTestSplit(),
    RenameFieldsCTICH(),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(CT_ICH)
