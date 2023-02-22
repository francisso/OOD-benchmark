import numpy as np
from amid import CacheToDisk
from amid.lidc import LIDC as AmidLIDC
from connectome import Chain, Filter, Transform, Apply, CacheToRam
from sklearn.model_selection import train_test_split

from ...const import RANDOM_STATE, TEST_SIZE_CT, CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TumorCenters


__all__ = ['lidc', 'lidc_train_ids', 'lidc_test_ids', ]


class RenameFieldsLIDC(Transform):
    __inherit__ = True

    def mask(cancer):
        return cancer

    def spacing(voxel_spacing):
        return voxel_spacing


lidc = Chain(
    AmidLIDC(),
    RenameFieldsLIDC(),
    Filter(lambda mask: not np.any(mask)),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    TumorCenters(),
    CacheToDisk(('ids', 'tumor_centers')),
    Apply(image=np.float16, mask=np.bool_),
    CacheToRam(),
    Apply(image=np.float32, mask=np.float32)
)


lidc_train_ids, lidc_test_ids = train_test_split(lidc.ids, test_size=TEST_SIZE_CT, random_state=RANDOM_STATE,
                                                 stratify=list(map(lidc.n_tumors, lidc.ids)))
