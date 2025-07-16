import awkward as ak
import numpy as np

from ucbtagfileprep import schema

def convert_jets_to_numpy(jet_pt, jet_eta, jet_phi, jet_energy, jet_mass,
                          jet_flavour, jet_dr, jet_is_matched, jet_flavour_label=None):
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
    jet_flavour_label : awkward.Array, optional
        Jet flavour label.

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
    #jet_data['flavour_label'] = ak.flatten(jet_flavour_label) if jet_flavour_label is not None else np.zeros(njet, dtype=np.int32)
    jet_data['dr'] = ak.flatten(jet_dr)
    jet_data['is_matched'] = ak.flatten(jet_is_matched)

    return jet_data

def convert_consts_to_numpy(track_valid, track_charge, track_d0, track_eta, track_phi, track_eta_rel, 
                            track_phi_rel, track_frac_pt, track_dr, track_z0, track_signed_2d_ip):
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
    consts_data['d0'] = ak.flatten(track_d0, axis=1)
    consts_data['eta'] = ak.flatten(track_eta, axis=1)
    consts_data['phi'] = ak.flatten(track_phi, axis=1)
    consts_data['eta_rel'] = ak.flatten(track_eta_rel, axis=1)
    consts_data['phi_rel'] = ak.flatten(track_phi_rel, axis=1)
    consts_data['frac_pt'] = ak.flatten(track_frac_pt, axis=1)
    consts_data['dr'] = ak.flatten(track_dr, axis=1)
    consts_data['z0'] = ak.flatten(track_z0, axis=1)
    consts_data['signed_2d_ip'] = ak.flatten(track_signed_2d_ip, axis=1)
    
    return consts_data
