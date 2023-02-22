import numpy as np
from amid import CacheToDisk
from amid.crossmoda import CrossMoDA as AmidCrossMoDA
from connectome import Chain, Filter, Transform, Apply

from ...const import MRI_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityMRI, AddShape, CanonicalMRIOrientation


__all__ = ['CrossMoDA', 'CrossMoDA_TEST_IDS', ]


class RenameFieldsCrossMoDA(Transform):
    __inherit__ = True

    def mask(masks):
        return masks == 1

    def spacing(pixel_spacing):
        return pixel_spacing


CrossMoDA = Chain(
    AmidCrossMoDA(),
    Filter(lambda id, split: split == 'training_source' and id.split('_')[1] == 'etz'),
    RenameFieldsCrossMoDA(),
    CanonicalMRIOrientation(),
    Rescale(new_spacing=MRI_COMMON_SPACING),
    ScaleIntensityMRI(),
    AddShape(),
    CacheToDisk('ids'),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)


CrossMoDA_TEST_IDS = CrossMoDA.ids
