import awkward as ak
import numpy as np

from ucbtagfileprep import kinematics

def match_obj1_to_obj2(
        obj1_eta, obj1_phi, obj2_eta, obj2_phi):
    """
    Match obj1 to the closest obj2 in eta,phi space. The functions to perform
    this are described in:

      https://awkward-array.org/doc/main/user-guide/how-to-combinatorics-best-match.html#combinations-with-nested-true


    Two objects are returned:
     - The index of the closest obj2.
     - The dR to the closest obj2.
    """

    # Helpful structures
    obj1 = ak.zip({'eta':obj1_eta,'phi':obj1_phi})
    obj2 = ak.zip({'eta':obj2_eta,'phi':obj2_phi})

    # Nested cartesian product matches all jets with truth particles
    obj1,obj2 = ak.unzip(ak.cartesian([obj1, obj2], nested=True))

    # Calculate the dR between jets and truth particles
    dr = kinematics.dr(obj1['eta'], obj1['phi'], obj2['eta'], obj2['phi'])

    # Get the index of the best match and store the corresponding dR
    bestMatch = ak.argmin(dr, axis=-1, keepdims=True)
    bestdR = ak.min(dr, axis=-1)

    # Unflatten the last axis to match the indices with showerData
    bestMatch=ak.flatten(bestMatch, axis=-1)

    return bestMatch, bestdR

def match_jets_to_quarks(jet_eta, jet_phi,
                         mc_eta, mc_phi, mc_pdgid,
                         dr_threshold=0.4):
    """
    Match jets to the closest quark using the logic of match_obj1_to_obj2.

    Parameters
    ----------
    jet_eta : awkward.Array
        Jet eta values
    jet_phi : awkward.Array
        Jet phi values
    mc_eta : awkward.Array
        Monte Carlo particle eta values
    mc_phi : awkward.Array
        Monte Carlo particle phi values
    mc_pdgid : awkward.Array
        Monte Carlo particle PDG IDs
    dr_threshold : float, optional
        Maximum dR for a valid match (default: 0.4)

    Returns
    -------
    jet_flavour : awkward.Array
        The flavour of the jet matched to the closest truth particle.
    jet_mcdr : awkward.Array
        The dR to the closest truth particle.
    jet_ismatched : awkward.Array
        A boolean array indicating if the jet is matched to a truth particle.
    """
    bestMatch, bestdR = match_obj1_to_obj2(jet_eta, jet_phi, mc_eta, mc_phi)

    # Check if the match is within the dR threshold
    jet_ismatched = bestdR < dr_threshold

    # Get the flavour (absolute value of PDG ID) for matched jets
    jet_flavour = ak.where(jet_ismatched, np.abs(mc_pdgid[bestMatch]), -1)

    # Set dR to a large value for unmatched jets
    jet_mcdr = ak.where(jet_ismatched, bestdR, 999.0)

    return jet_flavour, jet_mcdr, jet_ismatched, bestMatch
