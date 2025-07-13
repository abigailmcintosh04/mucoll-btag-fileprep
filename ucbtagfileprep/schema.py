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
