import numpy as np
from typing import Union

from .lidc import LIDC, PathLike
from ..augmentations import aug_list
from ...const import RANDOM_STATE


def decode_id(i):
    base_id, aug = i.split("_")
    aug_name, scale = aug.split(":")
    scale = float(scale)
    return base_id, aug_name, scale


__all__ = ['LIDC_AUG', decode_id]


class LIDC_AUG(LIDC):
    
    __all_params = {
        "elastic.transform": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
        "blur.transform": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .99],
        "slicedrop.transform": np.round(np.arange(0.05, 0.40001, 0.05), 2),
        "contrast.transform": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "corruption.transform": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "pixelshuffling.transform": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .99],
    }
    
    def __init__(self, root: Union[PathLike, None] = None):
        super().__init__(root)

        augmentations_list = [(k, v) for k in LIDC_AUG.__all_params
                                   for v in LIDC_AUG.__all_params[k]]
        
        # turn into  {previous id}_corruption.transform:0.4 0 like
        self.test_ids = sorted(tuple([f"{i}_{k}:{v}"
                                      for i in self.test_ids
                                      for (k,v) in augmentations_list]))
        self.train_ids = []
        
    def __getattr__(self, name):
        if name == "ids":
            return getattr(self._shadowed, name)

        def casted_to_original_id(i):
            # this will work both for original ids and modified ones
            base_id = i.split("_")[0]
            return getattr(self._shadowed, name)(base_id)

        return casted_to_original_id
    
    def image(self, i, debug=False):
        base_id, aug_name, scale = decode_id(i)
        base_image = self._shadowed.image(base_id)
        transformed = aug_list[aug_name](base_image, param=scale, random_state=RANDOM_STATE)
        if debug:
            print(aug_name, scale)
            return transformed, base_image
        return transformed

    def mask(self, i, debug=False):
        """
        Change mask only for elastic.transform augmentation, since it physically changes object positions
        """
        base_id, aug_name, scale = decode_id(i)
        base_mask = self._shadowed.mask(base_id)
        if aug_name == "elastic.transform":
            transformed = aug_list[aug_name](img=None, mask=base_mask, param=scale, random_state=RANDOM_STATE)
        else:
            transformed = base_mask

        if debug:
            print(aug_name, scale)
            return transformed, base_mask

        return transformed
