from typing import Union

import numpy as np
from amid import CacheToDisk
from amid.lidc import LIDC
from connectome import Filter, Transform, Apply, CacheToRam, chained, meta, CacheColumns
from sklearn.model_selection import train_test_split

from ...const import RANDOM_STATE, TEST_SIZE_CT, CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TumorCenters


__all__ = ['LIDC', ]


class RenameFieldsLIDC(Transform):
    __inherit__ = True

    def mask(cancer):
        return cancer

    def spacing(voxel_spacing):
        return voxel_spacing


class TrainTestSplitLIDC(Transform):
    __inherit__ = True

    _test_size: float
    _random_state: Union[int, None]

    def _train_test_split(ids, _test_size, _random_state, n_tumors):
        return train_test_split(ids, test_size=_test_size, random_state=_random_state,
                                stratify=list(map(n_tumors, ids)))

    @meta
    def train_ids(_train_test_split):
        return _train_test_split[0]

    @meta
    def test_ids(_train_test_split):
        return _train_test_split[1]


LIDC = chained(
    RenameFieldsLIDC(),
    Filter(lambda mask: not np.any(mask)),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    TumorCenters(),
    TrainTestSplitLIDC(test_size=TEST_SIZE_CT, random_state=RANDOM_STATE),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('tumor_centers', 'n_tumors', 'shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    CacheToRam(),
    Apply(image=np.float32, mask=np.float32)
)(LIDC)


# lidc_train_ids, lidc_test_ids = train_test_split(lidc.ids, test_size=TEST_SIZE_CT, random_state=RANDOM_STATE,
#                                                  stratify=list(map(lidc.n_tumors, lidc.ids)))
