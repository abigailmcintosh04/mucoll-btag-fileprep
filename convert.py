#!/usr/bin/env python

# %% Important imports

import argparse

from ucbtagfileprep import kinematics
from ucbtagfileprep import convert
from ucbtagfileprep import match
from ucbtagfileprep import tracks

import uproot
import h5py
import awkward as ak

# %% Input argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input file')
parser.add_argument('output_file', type=str, help='Path to the output file')

args = parser.parse_args()

input_path = args.input_file
output_path = args.output_file

# %% Read the input file
fh_in=uproot.open(input_path)

#
# Read the reconstructed jets

# Uproot can only load certain branches. Not clear why.
keys=fh_in['BUVertices'].keys()
keys.remove('evpro')
BUVertices=fh_in['BUVertices'].arrays(keys)

# Calculate jet kinematics
BUVertices['jmot'] = kinematics.pt   (BUVertices['jmox'], BUVertices['jmoy'])
BUVertices['jphi'] = kinematics.phi  (BUVertices['jmox'], BUVertices['jmoy'])
BUVertices['jthe'] = kinematics.theta(BUVertices['jmot'], BUVertices['jmoz'])
BUVertices['jeta'] = kinematics.eta  (BUVertices['jthe'])

#
# Read the truth particles

showerData = fh_in["showerData"]

# List required branches
branchsuffixes = ["mcPDGID", "mcE", "mcPx", "mcPy", "mcPz"]
branches = [f'd1_{suffix}' for suffix in branchsuffixes]
branches += [f'd2_{suffix}' for suffix in branchsuffixes]

# Read only the specified event range
showerData = showerData.arrays(branches)

# Unflatten the data
showerData = ak.unflatten(showerData, counts=1)
showerData = ak.Array({suffix : ak.concatenate([showerData[f'd1_{suffix}'], showerData[f'd2_{suffix}']], axis=1) for suffix in branchsuffixes})

# Calculate truth particle kinematics
showerData['mcPt'] = kinematics.pt(showerData['mcPx'], showerData['mcPy'])
showerData['mcPhi'] = kinematics.phi(showerData['mcPx'], showerData['mcPy'])
showerData['mcTheta'] = kinematics.theta(showerData['mcPt'], showerData['mcPz'])
showerData['mcEta'] = kinematics.eta(showerData['mcTheta'])

#
# Match the jets to the truth particles
BUVertices['jflv'], BUVertices['jmdr'], BUVertices['jism'] = match.match_jets_to_quarks(
    jet_eta=BUVertices['jeta'],
    jet_phi=BUVertices['jphi'],
    mc_eta=showerData['mcEta'],
    mc_phi=showerData['mcPhi'],
    mc_pdgid=showerData['mcPDGID']
)

#
# Handle the tracks
BUVertices['daughters_trackQ'] = tracks.charge(BUVertices['daughters_trackOmega'])
BUVertices['daughters_trackTheta'] = tracks.theta(BUVertices['daughters_trackTanLambda'])
BUVertices['daughters_trackPt'] = tracks.pt(BUVertices['daughters_trackOmega'])
BUVertices['daughters_trackEta'] = tracks.eta(BUVertices['daughters_trackTheta'])
BUVertices['daughters_trackValid'] = tracks.valid(BUVertices['daughters_trackOmega'])

#
# Prepare the jets output structures
jets = convert.convert_jets_to_numpy(
    jet_pt=BUVertices['jmot'],
    jet_eta=BUVertices['jeta'],
    jet_phi=BUVertices['jphi'],
    jet_energy=BUVertices['jene'],
    jet_mass=BUVertices['jmas'],
    jet_flavour=BUVertices['jflv'],
    jet_dr=BUVertices['jmdr'],
    jet_is_matched=BUVertices['jism']
)

print(dir(BUVertices))
consts = convert.convert_consts_to_numpy(
    track_valid=BUVertices['daughters_trackValid'],
    track_charge=BUVertices['daughters_trackQ'],
    track_d0=BUVertices['daughters_trackD0']
)

#
# Save to an H5 file

with h5py.File(output_path, 'w') as fh_out:
    fh_out.create_dataset('jets', data=jets)
