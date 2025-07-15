import numpy as np

# Dataset 'jets'
dtype_jets = np.dtype([
    ("pt", np.float32),
    ("eta", np.float32),
    ("phi", np.float32),
    ("energy", np.float32),
    ("mass", np.float32),
    ("flavour", np.int32),
    ("flavour_label", np.int32),
    ("dr", np.float32),
    ("is_matched", np.bool_),
])

# Dataset 'consts'
dtype_consts = np.dtype([
    ("valid", np.bool_),
    ("charge", np.int32),
    ("d0", np.float32),
    ('eta', np.float32),
])
