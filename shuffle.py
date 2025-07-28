import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
args = parser.parse_args()

input_path = args.input_file
output_path = args.output_file

path = 'tuples_small.h5'

with h5py.File(input_path, 'r') as h5file:
    jets = h5file['jets'][:]
    consts = h5file['consts'][:]

valid_mask = jets['is_matched'] == True
valid_jets = jets[valid_mask]

length = len(valid_jets)
indices = np.arange(0, length, 1)
np.random.shuffle(indices)

shuffled_tracks = consts[indices]
shuffled_jets = valid_jets[indices]

with h5py.File(output_path, 'w') as h5file:
    h5file['jets'] = shuffled_jets
    h5file['consts'] = shuffled_tracks