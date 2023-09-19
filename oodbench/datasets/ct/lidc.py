import numpy as np
from amid.internals import CacheToDisk, CacheColumns
from amid.lidc import LIDC as LIDC_AMID, Rescale, CanonicalCTOrientation
from connectome import Filter, Transform, Apply, CacheToRam, chained
from sklearn.model_selection import train_test_split

from ..transforms import AddShape, Identity, ScaleIntensityCT, TumorCenters
from ..wrappers import Proxy
from ...config import PATH_LIDC_RAW, USE_CACHING
from ...const import RANDOM_STATE, TEST_SIZE_CT, CT_COMMON_SPACING
from ...typing import OptPathLike


__all__ = ['LIDC', ]


class RenameFieldsLIDC(Transform):
    __inherit__ = True

    def mask(cancer):
        return cancer


class LIDC(Proxy):
    def __init__(self, root: OptPathLike = PATH_LIDC_RAW, use_caching: bool = USE_CACHING):
        dataset_chained = chained(
            CanonicalCTOrientation(),
            Rescale(new_spacing=CT_COMMON_SPACING),
            RenameFieldsLIDC(),
            Filter(lambda mask: np.any(mask)),
            ScaleIntensityCT(),
            AddShape(),
            TumorCenters(),
            CacheToDisk('ids') if use_caching else Identity(),
            CacheColumns(('tumor_centers', 'n_tumors', 'shape', 'spacing',)) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            CacheToRam(),
            Apply(image=np.float32, mask=np.float32)
        )(LIDC_AMID)

        super().__init__(dataset_chained(root))

        n_tumors = np.array(list(map(self.n_tumors, self.ids)))
        n_tumor_labels, n_tumor_counts = np.unique(n_tumors, return_counts=True)
        filter_labels = n_tumor_labels[n_tumor_counts <= 1]
        n_tumors_filter_mask = np.array([n not in filter_labels for n in n_tumors])

        n_tumors_grouped = n_tumors[n_tumors_filter_mask]
        ids_grouped, ids_unique = np.array(self.ids)[n_tumors_filter_mask], np.array(self.ids)[~n_tumors_filter_mask]

        train_ids, test_ids = train_test_split(ids_grouped, test_size=TEST_SIZE_CT, random_state=RANDOM_STATE,
                                               stratify=n_tumors_grouped)

        self.train_ids = sorted(tuple(train_ids.tolist() + ids_unique.tolist()))
        self.test_ids = sorted(tuple(test_ids.tolist()))
