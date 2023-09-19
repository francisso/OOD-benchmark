import numpy as np
from amid.egd import EGD as EGD_AMID
from amid.internals import CacheColumns, CacheToDisk
from connectome import Apply, Filter, Transform, chained

from ..transforms import AddShape, CanonicalOrientation, Identity, Rescale, ScaleIntensityMRI, TrainTestSplit
from ..wrappers import Proxy
from ...config import PATH_EGD_RAW, USE_CACHING
from ...const import MRI_COMMON_SPACING
from ...typing import OptPathLike


__all__ = ['EGD', ]


class RenameFieldsEGD(Transform):
    __inherit__ = True

    def mask(image):
        return np.zeros_like(image, dtype=bool)


class EGD(Proxy):
    def __init__(self, root: OptPathLike = PATH_EGD_RAW, use_caching: bool = USE_CACHING):
        dataset_chained = chained(
            Filter(lambda modality: modality == 'T1GD'),
            Filter(lambda field: field == 1.5),
            Filter(lambda manufacturer: manufacturer == 'SIEMENS'),
            TrainTestSplit(),
            RenameFieldsEGD(),
            CanonicalOrientation(),
            Rescale(new_spacing=MRI_COMMON_SPACING),
            ScaleIntensityMRI(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids',)) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing',)) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(EGD_AMID)

        super().__init__(dataset_chained(root))
