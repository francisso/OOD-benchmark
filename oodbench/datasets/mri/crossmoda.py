from typing import Union

import numpy as np
from amid.crossmoda import CrossMoDA as CrossMoDA_AMID
from amid.internals import CacheColumns, CacheToDisk
from connectome import Filter, Transform, Apply, chained

from ..transforms import AddShape, CanonicalOrientation, Identity, Rescale, ScaleIntensityMRI, TrainTestSplit
from ..wrappers import Proxy
from ...const import MRI_COMMON_SPACING
from ...typing import PathLike


__all__ = ['CrossMoDA', ]


class RenameFieldsCrossMoDA(Transform):
    __inherit__ = True

    def mask(masks):
        return masks == 1


class CrossMoDA(Proxy):
    def __init__(self, root: Union[PathLike, None] = None, use_caching: bool = True):
        dataset_chained = chained(
            Filter(lambda id: id.split('_')[1] == 'etz'),
            Filter(lambda split: split == 'training_source'),
            TrainTestSplit(),
            RenameFieldsCrossMoDA(),
            CanonicalOrientation(flip_x=False),
            Rescale(new_spacing=MRI_COMMON_SPACING),
            ScaleIntensityMRI(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids',)) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing',)) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(CrossMoDA_AMID)

        super().__init__(dataset_chained(root))
