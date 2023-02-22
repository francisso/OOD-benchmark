import numpy as np
from amid import CacheToDisk
from amid.vs_seg import VSSEG as AmidVSSEG, CanonicalMRIOrientation
from connectome import Chain, Filter, Transform, Apply, CacheToRam
from sklearn.model_selection import train_test_split

from ...const import RANDOM_STATE, TEST_SIZE, MRI_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityMRI, AddShape, TumorCenters


__all__ = ['VSSEG', 'VSSEG_TRAIN_IDS', 'VSSEG_TEST_IDS', ]


class RenameFieldsVSSEG(Transform):
    __inherit__ = True

    def mask(schwannoma):
        return schwannoma


VSSEG = Chain(
    AmidVSSEG(),
    Filter(lambda modality: modality == 'T1'),
    Filter(lambda meningioma: meningioma is None),
    Filter(lambda schwannoma: schwannoma is not None),
    CanonicalMRIOrientation(),
    RenameFieldsVSSEG(),
    Rescale(new_spacing=MRI_COMMON_SPACING),
    ScaleIntensityMRI(),
    AddShape(),
    TumorCenters(),
    CacheToDisk(('ids', 'tumor_centers')),
    Apply(image=np.float16, mask=np.bool_),
    CacheToRam(),
    Apply(image=np.float32, mask=np.float32)
)


VSSEG_TRAIN_IDS, VSSEG_TEST_IDS = train_test_split(VSSEG.ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
