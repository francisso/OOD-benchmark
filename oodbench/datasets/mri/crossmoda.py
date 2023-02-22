import numpy as np
from amid import CacheToDisk
from amid.crossmoda import CrossMoDA
from connectome import Chain, Filter, Transform, Apply

from ...const import MRI_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityMRI, AddShape, CanonicalMRIOrientation


__all__ = ['crossmoda', 'crossmoda_test_ids', ]


class RenameFieldsCrossMoDA(Transform):
    __inherit__ = True

    def mask(masks):
        return masks == 1

    def spacing(pixel_spacing):
        return pixel_spacing


crossmoda = Chain(
    CrossMoDA(),
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


crossmoda_test_ids = crossmoda.ids
