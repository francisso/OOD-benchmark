import numpy as np
from amid.internals import CacheToDisk, CacheColumns
from amid.lits import LiTS, CanonicalCTOrientation, Rescale
from connectome import Transform, Apply, chained

from ...const import CT_COMMON_SPACING
from ..transforms import ScaleIntensityCT, AddShape, TrainTestSplit


__all__ = ['LiTS', ]


class ChangeFieldsLiTS(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


LiTS = chained(
    TrainTestSplit(),
    CanonicalCTOrientation(),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ChangeFieldsLiTS(),
    ScaleIntensityCT(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(LiTS)
