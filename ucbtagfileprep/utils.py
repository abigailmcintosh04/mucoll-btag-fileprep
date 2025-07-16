import numpy as np

def prepare_lut(mymap):
    lookup = np.zeros(max(mymap.keys()) + 1, dtype=int)
    for k, v in mymap.items():
        lookup[k] = v
    return lookup
