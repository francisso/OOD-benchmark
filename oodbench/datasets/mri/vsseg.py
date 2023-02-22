import numpy as np
from amid import CacheToDisk
from amid.vs_seg import VSSEG, CanonicalMRIOrientation
from connectome import Chain, Filter, Transform, Apply, CacheToRam
from sklearn.model_selection import train_test_split

from ...const import RANDOM_STATE, TEST_SIZE_MRI, MRI_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityMRI, AddShape, TumorCenters


__all__ = ['vsseg', 'vsseg_train_ids', 'vsseg_test_ids', ]


class RenameFieldsVSSEG(Transform):
    __inherit__ = True

    def mask(schwannoma):
        return schwannoma


vsseg = Chain(
    VSSEG(),
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


vsseg_train_ids, vsseg_test_ids = train_test_split(vsseg.ids, test_size=TEST_SIZE_MRI, random_state=RANDOM_STATE)
