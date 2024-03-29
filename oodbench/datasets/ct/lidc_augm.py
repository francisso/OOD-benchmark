import ctypes

from .lidc import LIDC
from ..augmentations import AUGM_LIST, decode_id
from ...config import PATH_LIDC_RAW, USE_CACHING
from ...const import RANDOM_STATE
from ...typing import OptPathLike


__all__ = ['LIDC_AUGM', ]


class LIDC_AUGM(LIDC):

    __all_params = {
        "elastic.transform":   [0.1, 0.5, 1., 1.5, 2.0],
        "blur.transform":      [.1,  .3, .5, .7, .9],
        "slicedrop.transform": [.01, .75, .15, .25, .4],
        "contrast.transform":  [.1, .25, .4, .55, .7],
        "corruption.transform": [.1, .25, .4, .55, .7],
    }

    def __init__(self, root: OptPathLike = PATH_LIDC_RAW, use_caching: bool = USE_CACHING):
        super().__init__(root, use_caching)

        augmentations_list = [(k, v) for k in LIDC_AUGM.__all_params for v in LIDC_AUGM.__all_params[k]]

        # turn into  {previous id}:corruption.transform:0.4 0 like
        self.test_ids = sorted(tuple([f"{i}:{k}:{v}"
                                      for i in self.test_ids
                                      for (k, v) in augmentations_list]))
        self.train_ids = []

    def __getattr__(self, name):
        if name == "ids":
            return getattr(self._shadowed, name)

        def casted_to_original_id(i):
            base_id = i.split(':')[0]  # this will work both for original ids and modified ones!
            return getattr(self._shadowed, name)(base_id)

        return casted_to_original_id

    def image(self, i, debug=False):
        base_id, aug_name, scale = decode_id(i)
        base_image = self._shadowed.image(base_id)
        transformed = AUGM_LIST[aug_name](base_image, param=scale,
                                          random_state=ctypes.c_uint32(RANDOM_STATE + hash(i)).value)

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
            transformed = AUGM_LIST[aug_name](img=None, mask=base_mask, param=scale,
                                              random_state=ctypes.c_uint32(RANDOM_STATE + hash(i)).value)
        else:
            transformed = base_mask

        if debug:
            print(aug_name, scale)
            return transformed, base_mask

        return transformed
