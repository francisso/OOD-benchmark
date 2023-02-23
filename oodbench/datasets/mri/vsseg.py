import numpy as np
from amid import CacheToDisk
from amid.vs_seg import VSSEG, CanonicalMRIOrientation
from connectome import Filter, Transform, Apply, CacheToRam, chained, CacheColumns

from ...const import RANDOM_STATE, TEST_SIZE_MRI, MRI_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityMRI, AddShape, TumorCenters, TrainTestSplit


__all__ = ['VSSEG', ]


class RenameFieldsVSSEG(Transform):
    __inherit__ = True

    def mask(schwannoma):
        return schwannoma


VSSEG = chained(
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
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', 'tumor_centers', 'n_tumors', )),
    Apply(image=np.float16, mask=np.bool_),
    CacheToRam(),
    Apply(image=np.float32, mask=np.float32)
)(VSSEG)
