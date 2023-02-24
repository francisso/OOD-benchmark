import numpy as np
from amid import CacheToDisk, CacheColumns
from amid.crossmoda import CrossMoDA
from connectome import Filter, Transform, Apply, chained

from ...const import MRI_COMMON_SPACING
from ..transforms import Rescale, ScaleIntensityMRI, AddShape, CanonicalMRIOrientation, TrainTestSplit


__all__ = ['CrossMoDA', ]


class RenameFieldsCrossMoDA(Transform):
    __inherit__ = True

    def mask(masks):
        return masks == 1


CrossMoDA = chained(
    Filter(lambda id: id.split('_')[1] == 'etz'),
    Filter(lambda split: split == 'training_source'),
    TrainTestSplit(),
    RenameFieldsCrossMoDA(),
    CanonicalMRIOrientation(flip_x=False),
    Rescale(new_spacing=MRI_COMMON_SPACING),
    ScaleIntensityMRI(),
    AddShape(),
    CacheToDisk(('ids', 'train_ids', 'test_ids', )),
    CacheColumns(('shape', 'spacing', )),
    Apply(image=np.float16, mask=np.bool_),
    Apply(image=np.float32, mask=np.float32)
)(CrossMoDA)
