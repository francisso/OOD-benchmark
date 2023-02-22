import numpy as np
from amid import CacheToDisk
from amid.cc359 import CC359
from connectome import Chain, Transform, Apply

from ..transforms import ScaleIntensityMRI, AddShape, Rescale, CanonicalMRIOrientation
from ...const import MRI_COMMON_SPACING


__all__ = ['cc359', 'cc359_test_ids', ]


class RenameFieldsCC359(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)

    def spacing(voxel_spacing):
        return voxel_spacing


cc359 = Chain(
    CC359(),
    RenameFieldsCC359(),
    CanonicalMRIOrientation(),
    Rescale(new_spacing=MRI_COMMON_SPACING),
    ScaleIntensityMRI(),
    AddShape(),
    CacheToDisk('ids'),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)


cc359_test_ids = cc359.ids
