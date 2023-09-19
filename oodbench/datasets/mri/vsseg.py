from typing import Union

import numpy as np
from amid.internals import CacheColumns, CacheToDisk
from amid.vs_seg import VSSEG as VSSEG_AMID, CanonicalMRIOrientation
from connectome import Filter, Transform, Apply, CacheToRam, chained

from ..transforms import AddShape, Identity, Rescale, ScaleIntensityMRI, TrainTestSplit, TumorCenters
from ..wrappers import Proxy
from ...const import RANDOM_STATE, TEST_SIZE_MRI, MRI_COMMON_SPACING
from ...typing import PathLike

__all__ = ['VSSEG', ]


class RenameFieldsVSSEG(Transform):
    __inherit__ = True

    def mask(schwannoma):
        return schwannoma


class VSSEG(Proxy):
    def __init__(self, root: Union[PathLike, None] = None, use_caching: bool = True):
        dataset_chained = chained(
            Filter(lambda modality: modality == 'T1'),
            Filter(lambda meningioma: meningioma is None),
            Filter(lambda schwannoma: schwannoma is not None),
            TrainTestSplit(test_size=TEST_SIZE_MRI, random_state=RANDOM_STATE),
            CanonicalMRIOrientation(),
            RenameFieldsVSSEG(),
            Rescale(new_spacing=MRI_COMMON_SPACING),
            ScaleIntensityMRI(),
            AddShape(),
            TumorCenters(),
            CacheToDisk(('ids', 'train_ids', 'test_ids', )) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing', 'tumor_centers', 'n_tumors', )) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            CacheToRam(),
            Apply(image=np.float32, mask=np.float32)
        )(VSSEG_AMID)

        super().__init__(dataset_chained(root))
