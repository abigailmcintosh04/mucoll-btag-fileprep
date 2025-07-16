import awkward as ak
import numpy as np

from ucbtagfileprep import schema
from ucbtagfileprep import utils


flavour_labels_lut = utils.prepare_lut({1:0, 2:0, 3:0, 4:1, 5:2})

def convert_jets_to_numpy(jet_pt, jet_eta, jet_phi, jet_energy, jet_mass,
                          jet_flavour, jet_dr, jet_is_matched):
    """
    Convert jets data from an awkward array to a structured numpy array.

    Parameters
    ----------
    jet_pt : awkward.Array
        Jet transverse momentum.
    jet_eta : awkward.Array
        Jet pseudorapidity.
    jet_phi : awkward.Array
        Jet azimuthal angle.
    jet_energy : awkward.Array
        Jet energy.
    jet_mass : awkward.Array
        Jet mass.
    jet_flavour : awkward.Array
        Jet flavour.
    jet_dr : awkward.Array
        Jet distance in delta R.
    jet_is_matched : awkward.Array
        Boolean array indicating if jet is matched to truth particle.

    Returns
    -------
    numpy.ndarray
        A structured numpy array with jet information.
    """

    njet=ak.count(jet_pt)

    jet_data = np.empty(njet, dtype=schema.dtype_jets)
    jet_data['pt'] = ak.flatten(jet_pt)
    jet_data['eta'] = ak.flatten(jet_eta)
    jet_data['phi'] = ak.flatten(jet_phi)
    jet_data['energy'] = ak.flatten(jet_energy)
    jet_data['mass'] = ak.flatten(jet_mass)
    jet_data['flavour'] = ak.flatten(jet_flavour)
    jet_data['flavour_label'] = flavour_labels_lut[jet_data['flavour']]
    jet_data['dr'] = ak.flatten(jet_dr)
    jet_data['is_matched'] = ak.flatten(jet_is_matched)

    return jet_data

def convert_consts_to_numpy(track_valid, track_charge, track_phi_rel, track_eta_rel, track_d0):
    """
    Convert consts data from an awkward array to a structured numpy array.

    Parameters
    ----------
    track_valid : awkward.Array
        The track is real and not a dummy filler.
    track_charge : awkward.Array
        Track charge.
    track_d0 : awkward.Array
        Track d0.

    Returns
    -------
    numpy.ndarray
        A structured numpy array with consts information.
    """

    # Save the arrays
    ntracks = np.sum(ak.num(track_valid, axis=1))

    consts_data = np.empty((ntracks, 200), dtype=schema.dtype_consts)
    consts_data['valid'] = ak.flatten(track_valid, axis=1)
    consts_data['charge'] = ak.flatten(track_charge, axis=1)
    consts_data['phi_rel'] = ak.flatten(track_phi_rel, axis=1)
    consts_data['eta_rel'] = ak.flatten(track_eta_rel, axis=1)
    consts_data['d0'] = ak.flatten(track_d0, axis=1)

    return consts_data
