from pathlib import Path
from typing import Union

import numpy as np

from amid import CacheToDisk, CacheColumns
from amid.lidc import LIDC as LIDCAmid
from connectome import Filter, Transform, Apply, CacheToRam, chained
from sklearn.model_selection import train_test_split

from ...const import RANDOM_STATE, TEST_SIZE_CT, CT_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityCT, AddShape, TumorCenters
from ..wrappers import Proxy


__all__ = ['LIDC', ]


PathLike = Union[Path, str]


class RenameFieldsLIDC(Transform):
    __inherit__ = True

    def mask(cancer):
        return cancer

    def spacing(voxel_spacing):
        return voxel_spacing


LIDCChained = chained(
    RenameFieldsLIDC(),
    Filter(lambda mask: not np.any(mask)),
    Rescale(new_spacing=CT_COMMON_SPACING),
    ScaleIntensityCT(),
    AddShape(),
    TumorCenters(),
    CacheToDisk('ids'),
    CacheColumns(('tumor_centers', 'n_tumors', 'shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    CacheToRam(),
    Apply(image=np.float32, mask=np.float32)
)(LIDCAmid)


class LIDC(Proxy):
    def __init__(self, root: Union[PathLike, None] = None):
        super().__init__(LIDCChained(root))
        self.train_ids, self.test_ids = train_test_split(self.ids, test_size=TEST_SIZE_CT, random_state=RANDOM_STATE,
                                                         stratify=list(map(self.n_tumors, self.ids)))
