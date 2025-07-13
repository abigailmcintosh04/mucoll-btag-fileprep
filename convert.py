#!/usr/bin/env python

# %% Important imports
import uproot

import argparse

from ucbtagfileprep import kinematics

# %% Input argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input file')
parser.add_argument('output_file', type=str, help='Path to the output file')

args = parser.parse_args()

input_path = args.input_file
output_path = args.output_file

# %% Read the input file
fh_in=uproot.open(input_path)

# Uproot can only load certain branches. Not clear why.
keys=fh_in['BUVertices'].keys()
keys.remove('evpro')
BUVertices=fh_in['BUVertices'].arrays(keys)

print(BUVertices)

# %% Calculate jet kinematics
BUVertices['jmot'],BUVertices['jeta'],BUVertices['jphi'] = \
    kinematics.ptetaphi(BUVertices['jmox'],BUVertices['jmoy'],BUVertices['jmoz'])

