import numpy as np
from amid import CacheToDisk, CacheColumns
from amid.egd import EGD
from connectome import Transform, Apply, Filter, chained

from ..transforms import ScaleIntensityMRI, AddShape, Rescale, CanonicalMRIOrientation, TrainTestSplit
from ...const import MRI_COMMON_SPACING


__all__ = ['EGD', ]


class RenameFieldsEGD(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


EGD = chained(
    Filter(lambda modality: modality == 'T1GD'),
    Filter(lambda field: field == 1.5),
    Filter(lambda manufacturer: manufacturer == 'SIEMENS'),
    TrainTestSplit(),
    RenameFieldsEGD(),
    CanonicalMRIOrientation(),
    Rescale(new_spacing=MRI_COMMON_SPACING),
    ScaleIntensityMRI(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(EGD)
