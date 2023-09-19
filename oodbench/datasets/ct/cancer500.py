import numpy as np
from amid.cancer_500 import MoscowCancer500
from amid.internals import CacheToDisk, CacheColumns
from connectome import Apply, Filter, Transform, chained
from connectome.interface.nodes import Output

from ..transforms import AddShape, CanonicalOrientation, Identity, Rescale, ScaleIntensityCT, TrainTestSplit
from ..wrappers import Proxy
from ...config import PATH_CANCER500_RAW, USE_CACHING
from ...const import CT_COMMON_SPACING
from ...typing import OptPathLike


__all__ = ['Cancer500', ]


class CreateFieldsCancer500(Transform):
    __inherit__ = True

    _mask_hu_threshold: int = -500

    def mask(nodules, image, spacing: Output, _mask_hu_threshold):
        mask = np.zeros_like(image, dtype=bool)
        if nodules is not None:
            for nodule in nodules:
                center_sum = 0
                diameters = []
                for ann_nodule in nodule.values():
                    center_sum += np.array(ann_nodule.center_voxel)
                    diameters.append(ann_nodule.diameter_mm)

                center = np.round(center_sum / len(nodule))
                diameters = np.array(diameters, dtype=float)
                if np.isnan(diameters).all():
                    continue

                diameter = np.nanmean(diameters)

                # create circular mask
                c, s = center, spacing
                x, y, z = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
                dist_from_center = np.sqrt(s[0] * (x - c[0]) ** 2 + s[1] * (y - c[1]) ** 2 + s[2] * (z - c[2]) ** 2)
                current_mask = dist_from_center <= (diameter / 2)

                mask |= current_mask

        mask &= (image > _mask_hu_threshold)

        return mask

    def spacing(pixel_spacing, slice_locations):
        return *pixel_spacing, np.diff(slice_locations).mean().item()


def filter_slice_locations(slice_locations, n_slices_min: int = 64, rel_delta_max: float = 1e-2):
    if len(slice_locations) < n_slices_min:
        return False
    diffs = np.diff(slice_locations)
    if (diffs.max() - diffs.min()) / diffs.mean() > rel_delta_max:
        return False
    return True


class Cancer500(Proxy):
    def __init__(self, root: OptPathLike = PATH_CANCER500_RAW, use_caching: bool = USE_CACHING):
        dataset_chained = chained(
            Filter(lambda slice_locations: filter_slice_locations(slice_locations)),
            TrainTestSplit(),
            CreateFieldsCancer500(),
            CanonicalOrientation(flip_x=False),
            Rescale(new_spacing=CT_COMMON_SPACING),
            ScaleIntensityCT(),
            AddShape(),
            CacheToDisk(('ids', 'train_ids', 'test_ids',)) if use_caching else Identity(),
            CacheColumns(('shape', 'spacing',)) if use_caching else Identity(),
            Apply(image=np.float16, mask=np.bool_),
            Apply(image=np.float32, mask=np.float32)
        )(MoscowCancer500)

        super().__init__(dataset_chained(root))
