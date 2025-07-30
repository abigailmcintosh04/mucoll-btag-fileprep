#!/usr/bin/env python

# Important imports

import argparse

from ucbtagfileprep import kinematics
from ucbtagfileprep import convert
from ucbtagfileprep import match
from ucbtagfileprep import tracks

import uproot
import h5py
import awkward as ak
import numpy as np

# Input argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input file')
parser.add_argument('output_file', type=str, help='Path to the output file')
parser.add_argument('shuffle', type=bool, help='Output file shuffled or not.')

args = parser.parse_args()

input_path = args.input_file
output_path = args.output_file
shuffle = args.shuffle

# Read the input file
fh_in=uproot.open(input_path)

#
# Read the reconstructed jets
# Uproot can only load certain branches. Not clear why.
keys=fh_in['JET_kt'].keys()
keys.remove('evpro')
keys.remove('vttyp')
JET_kt=fh_in['JET_kt'].arrays(keys)

#
# Read the truth particles
showerData = fh_in["showerData"]

#
# Read the truth jets
keys = fh_in['TrueJets'].keys()
keys.remove('evpro')
TrueJets = fh_in['TrueJets'].arrays(keys)

#
# Calculate jet kinematics
JET_kt['jmot'] = kinematics.pt   (JET_kt['jmox'], JET_kt['jmoy'])
JET_kt['jphi'] = kinematics.phi  (JET_kt['jmox'], JET_kt['jmoy'])
JET_kt['jthe'] = kinematics.theta(JET_kt['jmot'], JET_kt['jmoz'])
JET_kt['jeta'] = kinematics.eta  (JET_kt['jthe'])

#
# Calculate truth jet kinematics
TrueJets['jmot'] = kinematics.pt(TrueJets['jmox'], TrueJets['jmoy'])
TrueJets['jphi'] = kinematics.phi(TrueJets['jmox'], TrueJets['jmoy']) 
TrueJets['jthe'] = kinematics.theta(TrueJets['jmot'], TrueJets['jmoz'])
TrueJets['jeta'] = kinematics.eta(TrueJets['jthe'])

# print(len(JET_kt['jmot']))
# print(len(TrueJets['jmot']))
# print('-----------------------')

#
# List required branches
branchsuffixes = ["mcPDGID", "mcE", "mcPx", "mcPy", "mcPz"]
branches = [f'd1_{suffix}' for suffix in branchsuffixes]
branches += [f'd2_{suffix}' for suffix in branchsuffixes]

#
# Read only the specified event range
showerData = showerData.arrays(branches)

#
# Unflatten the data
showerData = ak.unflatten(showerData, counts=1)
showerData = ak.Array({suffix : ak.concatenate([showerData[f'd1_{suffix}'], showerData[f'd2_{suffix}']], axis=1) for suffix in branchsuffixes})

#
# Calculate truth particle kinematics
showerData['mcPt'] = kinematics.pt(showerData['mcPx'], showerData['mcPy'])
showerData['mcPhi'] = kinematics.phi(showerData['mcPx'], showerData['mcPy'])
showerData['mcTheta'] = kinematics.theta(showerData['mcPt'], showerData['mcPz'])
showerData['mcEta'] = kinematics.eta(showerData['mcTheta'])

# print(len(JET_kt['jeta']))
# print(len(JET_kt['jphi']))
# print(len(showerData['mcEta']))
# print(len(showerData['mcPhi']))
# print(len(showerData['mcPDGID']))
# print(len(TrueJets['jmot']))

#
# Match the jets to the truth particles
JET_kt['jflv'], JET_kt['jmdr'], JET_kt['jism'] = match.match_jets_to_quarks(
    jet_eta=JET_kt['jeta'],
    jet_phi=JET_kt['jphi'],
    mc_eta=showerData['mcEta'],
    mc_phi=showerData['mcPhi'],
    mc_pdgid=showerData['mcPDGID'],
    # mc_pt=showerData['mcPt']
)

JET_kt['jtpt'] = match.match_jets_to_truthjets(
    jet_eta=JET_kt['jeta'],
    jet_phi=JET_kt['jphi'],
    truth_eta=TrueJets['jeta'],
    truth_phi=TrueJets['jphi'],
    truth_pt=TrueJets['jmot']
)

#
# Handle the tracks
JET_kt['daughters_trackQ'] = tracks.charge(JET_kt['daughters_trackOmega'])
JET_kt['daughters_trackTheta'] = tracks.theta(JET_kt['daughters_trackTanLambda'])
JET_kt['daughters_trackPt'] = tracks.pt(JET_kt['daughters_trackOmega'])
JET_kt['daughters_trackEta'] = tracks.eta(JET_kt['daughters_trackTheta'])
JET_kt['daughters_trackValid'] = tracks.valid(JET_kt['daughters_trackOmega'])
JET_kt['daughters_trackPhiRel'] = tracks.phi_rel(JET_kt['jphi'], JET_kt['daughters_trackPhi'], JET_kt['daughters_trackValid'])
JET_kt['daughters_trackEtaRel'] = tracks.eta_rel(JET_kt['jeta'], JET_kt['daughters_trackEta'], JET_kt['daughters_trackValid'])
JET_kt['daughters_trackPtFrac'] = JET_kt['daughters_trackPt'] / JET_kt['jmot']
JET_kt['daughters_trackdR'] = tracks.deltaR(JET_kt['daughters_trackPhiRel'], JET_kt['daughters_trackEtaRel'])
JET_kt['daughters_track2DIP'] = tracks.signed_2d_ip(JET_kt['daughters_trackD0'], JET_kt['daughters_trackSigmaD0'], JET_kt['daughters_trackPhiRel'], JET_kt['daughters_trackValid'])
JET_kt['daughters_track3DIP'] = tracks.signed_3d_ip(JET_kt['daughters_trackD0'], JET_kt['daughters_trackZ0'], JET_kt['daughters_trackSigmaD0'], JET_kt['daughters_trackSigmaZ0'], JET_kt['daughters_trackPhiRel'], JET_kt['daughters_trackValid'])

#
# Prepare the jets output structures
jets = convert.convert_jets_to_numpy(
    jet_truth_pt = JET_kt['jtpt'],
    jet_pt = JET_kt['jmot'],
    jet_eta = JET_kt['jeta'],
    jet_phi = JET_kt['jphi'],
    jet_energy = JET_kt['jene'],
    jet_mass = JET_kt['jmas'],
    jet_flavour = JET_kt['jflv'],
    jet_dr = JET_kt['jmdr'],
    jet_is_matched = JET_kt['jism']
)

consts = convert.convert_consts_to_numpy(
    track_valid = JET_kt['daughters_trackValid'],
    track_charge = JET_kt['daughters_trackQ'],
    track_d0 = JET_kt['daughters_trackD0'],
    track_eta = JET_kt['daughters_trackEta'],
    track_phi = JET_kt['daughters_trackPhi'],
    track_eta_rel = JET_kt['daughters_trackEtaRel'],
    track_phi_rel = JET_kt['daughters_trackPhiRel'],
    track_pt_frac = JET_kt['daughters_trackPtFrac'],
    track_dr = JET_kt['daughters_trackdR'],
    track_z0 = JET_kt['daughters_trackZ0'],
    track_signed_2d_ip = JET_kt['daughters_track2DIP'],
    track_signed_3d_ip = JET_kt['daughters_track3DIP'],
)

#
# Save to an H5 file

with h5py.File(output_path, 'w') as fh_out:
    if shuffle:
        valid_mask = jets['is_matched'] == True
        valid_jets = jets[valid_mask]
        length = len(valid_jets)
        indices = np.arange(0, length, 1)
        np.random.shuffle(indices)
        shuffled_consts = consts[indices]
        shuffled_jets = valid_jets[indices]
        jets = shuffled_jets
        consts = shuffled_consts
    fh_out.create_dataset('jets', data=jets)
    fh_out.create_dataset('consts', data=consts)
    fh_out['jets'].attrs['flavour_label'] = np.array(['ujets', 'cjets', 'bjets'], dtype=object)
