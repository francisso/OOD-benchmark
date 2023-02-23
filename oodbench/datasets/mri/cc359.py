import numpy as np
from amid import CacheToDisk, CacheColumns
from amid.cc359 import CC359
from connectome import Transform, Apply, chained

from ..transforms import ScaleIntensityMRI, AddShape, Rescale, CanonicalMRIOrientation, TrainTestSplit
from ...const import MRI_COMMON_SPACING


__all__ = ['CC359', ]


class RenameFieldsCC359(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)

    def spacing(voxel_spacing):
        return voxel_spacing


CC359 = chained(
    TrainTestSplit(),
    RenameFieldsCC359(),
    CanonicalMRIOrientation(),
    Rescale(new_spacing=MRI_COMMON_SPACING),
    ScaleIntensityMRI(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(CC359)
