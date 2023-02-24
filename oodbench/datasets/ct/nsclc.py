import numpy as np
from amid import CacheToDisk, CacheColumns
from amid.nsclc import NSCLC
from connectome import Apply, chained, Transform, Filter

from ...const import CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TrainTestSplit, CanonicalOrientation


__all__ = ['NSCLC', ]


class ChangeFieldsNSCLC(Transform):
    __inherit__ = True

    def mask(mask, image):
        return np.zeros_like(image, dtype=bool) if (mask is None) else (mask > 0.5)


NSCLC = chained(
    ChangeFieldsNSCLC(),
    Filter(lambda image: image.shape[-1] >= 64),
    Filter(lambda mask: np.any(mask)),
    TrainTestSplit(),
    CanonicalOrientation(flip_x=False),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(NSCLC)
