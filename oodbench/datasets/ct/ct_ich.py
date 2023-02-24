import numpy as np
from amid import CacheToDisk, CacheColumns
from amid.ct_ich import CT_ICH
from connectome import Transform, Apply, chained

from ...const import CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TrainTestSplit, CanonicalOrientation


__all__ = ['CT_ICH', ]


class RenameFieldsCTICH(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


CT_ICH = chained(
    TrainTestSplit(),
    RenameFieldsCTICH(),
    CanonicalOrientation(),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(CT_ICH)
