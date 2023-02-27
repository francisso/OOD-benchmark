import ctypes
from pathlib import Path
import numpy as np
from typing import Union

from .vsseg import VSSEG as VSSEGChained
from ..augmentations import aug_list
from ..wrappers import Proxy
from ...const import RANDOM_STATE


def decode_id(i):
    base_id, aug = i.split("_")
    aug_name, scale = aug.split(":")
    scale = float(scale)
    return base_id, aug_name, scale


__all__ = ["VSSEG_AUG", decode_id]

PathLike = Union[Path, str]


class VSSEG_AUG(Proxy):

    __all_params = {
        "slicedrop.transform": [.01, .75, .15, .25, .4],
        "contrast.transform":  [.1, .25, .4, .55, .7],
        "corruption.transform": [.1, .25, .4, .55, .7],
        "pixelshuffling.transform": [.1,  .3, .5, .7, .9],
        "ghosting.transform": [1, 2, 3, 4, 5],
        "anisotropy.transform": [1, 2, 3, 4, 5],
        "spike.transform": [1, 2, 3, 4, 5],
        "motion.transform": [1, 2, 3, 4, 5],
    }

    def __init__(self, root: Union[PathLike, None] = None):
        super().__init__(VSSEGChained(root))
        augmentations_list = [(k, v) for k in VSSEG_AUG.__all_params
                              for v in VSSEG_AUG.__all_params[k]]

        # turn into  {previous id}_corruption.transform:0.4 0 like
        self.test_ids = sorted(tuple([f"{i}_{k}:{v}"
                                      for i in self._shadowed.test_ids
                                      for (k, v) in augmentations_list]))
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
        transformed = aug_list[aug_name](base_image, param=scale,
                                         random_state=ctypes.c_uint32(RANDOM_STATE + hash(i)).value)
        if debug:
            print(aug_name, scale)
            return transformed, base_image
        return transformed