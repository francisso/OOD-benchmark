from .typing import OptPathLike


# Paths to raw CT dataset:
PATH_CANCER500_RAW: OptPathLike = None
PATH_CTICH_RAW: OptPathLike = None
PATH_LIDC_RAW: OptPathLike = None
PATH_LITS_RAW: OptPathLike = None
PATH_MEDSEG9_RAW: OptPathLike = None
PATH_MIDRC_RAW: OptPathLike = None

# Paths to raw MRI dataset:
PATH_CC359_RAW: OptPathLike = None
PATH_CROSSMODA_RAW: OptPathLike = None
PATH_EGD_RAW: OptPathLike = None
PATH_VSSEG_RAW: OptPathLike = None

# Option to speed-up datasets loading by caching lightweight fields, such as `ids`:
# (requires setting up AMID config!)
USE_CACHING: bool = True
