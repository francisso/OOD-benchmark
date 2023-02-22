import numpy as np
from amid import CacheToDisk
from amid.ct_ich import CT_ICH
from connectome import Chain, Transform, Apply

from ...const import CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape


__all__ = ['ct_ich', 'ct_ich_test_ids', ]


class RenameFieldsCTICH(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)

    def spacing(voxel_spacing):
        return voxel_spacing


ct_ich = Chain(
    CT_ICH(),
    RenameFieldsCTICH(),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    CacheToDisk('ids'),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)


ct_ich_test_ids = ct_ich.ids
