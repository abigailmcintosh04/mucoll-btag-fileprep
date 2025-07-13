import numpy as np
import awkward as ak

def pt(px, py):
    pt = (px**2 + py**2)**0.5

    return pt

def phi(px, py):
    phi = np.arctan2(py, px)

    return phi

def theta(pt, pz):
    theta = np.arctan2(pt, pz)

    return theta

def eta(theta):
    eta = -np.log(np.tan(theta / 2))

    return eta

def dr(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)

    return (deta**2 + dphi**2)**0.5

def delta_phi(phi1, phi2):
    """
    Calculate delta phi between two phi values.
    Returns delta phi in the range [-pi, pi].
    """
    dphi = phi1 - phi2

    # Use ak.where to handle the wrapping
    dphi = ak.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = ak.where(dphi < -np.pi, dphi + 2*np.pi, dphi)

    return dphi