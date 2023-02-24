import numpy as np
from amid.internals import CacheToDisk, CacheColumns
from amid.midrc import MIDRC
from connectome import Transform, Apply, chained, Filter

from ...const import CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TrainTestSplit, CanonicalOrientation


__all__ = ['MIDRC', ]


class ChangeFieldsMIDRC(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


MIDRC = chained(
    Filter(lambda mask: (mask is not None) and (np.any(mask[0]) or np.any(mask[-1]))),
    TrainTestSplit(),
    ChangeFieldsMIDRC(),
    CanonicalOrientation(flip_x=False),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(MIDRC)
