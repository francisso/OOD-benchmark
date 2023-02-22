import numpy as np
from amid import CacheToDisk
from amid.egd import EGD
from connectome import Chain, Transform, Apply, Filter

from ..transforms import ScaleIntensityMRI, AddShape, Rescale, CanonicalMRIOrientation
from ...const import MRI_COMMON_SPACING


__all__ = ['egd', 'egd_test_ids', ]


class RenameFieldsEGD(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)

    def spacing(voxel_spacing):
        return voxel_spacing


egd = Chain(
    EGD(),
    Filter(lambda modality: modality == 'T1GD'),
    Filter(lambda field: field == 1.5),
    Filter(lambda manufacturer: manufacturer == 'SIEMENS'),
    RenameFieldsEGD(),
    CanonicalMRIOrientation(),
    Rescale(new_spacing=MRI_COMMON_SPACING),
    ScaleIntensityMRI(),
    AddShape(),
    CacheToDisk('ids'),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)


egd_test_ids = egd.ids
