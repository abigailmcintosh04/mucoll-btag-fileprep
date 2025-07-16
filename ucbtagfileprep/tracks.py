import numpy as np
import awkward as ak
import kinematics

def charge(omega):
    """
    Calculate the charge of a track from its omega parameter.

    Parameters
    ----------
    omega : awkward.Array
        The omega parameter (curvature) of the track.

    Returns
    -------
    awkward.Array
        The charge of the track (+1 or -1).
    """
    # Charge is the sign of omega (curvature)
    return np.sign(omega)

def theta(tan_lambda):
    """
    Calculate the polar angle theta from tan(lambda).

    Parameters
    ----------
    tan_lambda : awkward.Array
        The tangent of the lambda angle (dip angle).

    Returns
    -------
    awkward.Array
        The polar angle theta in radians.
    """
    # theta = pi/2 - lambda, where lambda = arctan(tan_lambda)
    lambda_angle = np.arctan(tan_lambda)
    return np.pi/2 - lambda_angle

def pt(omega, B_field=3.57):
    """
    Calculate the transverse momentum from omega.

    Parameters
    ----------
    omega : awkward.Array
        The omega parameter (curvature) of the track.
    B_field : float, optional
        Magnetic field strength in Tesla (default: 3.57).

    Returns
    -------
    awkward.Array
        The transverse momentum.
    """
    return ak.where(omega != 0, (0.0003 * B_field) / np.abs(omega), 0.0)

def eta(theta):
    """
    Calculate pseudorapidity from polar angle theta.

    Parameters
    ----------
    theta : awkward.Array
        The polar angle theta in radians.

    Returns
    -------
    awkward.Array
        The pseudorapidity eta.
    """
    # eta = -ln(tan(theta/2))
    return -np.log(np.tan(theta / 2))

def valid(omega):
    """
    Check if a track is valid by verifying the curvature is not zero.

    Parameters
    ----------
    omega : awkward.Array
        The omega parameter (curvature) of the track.

    Returns
    -------
    awkward.Array
        Boolean array indicating if the track is valid (omega != 0).
    """
    return omega != 0

def phi_rel(jet_phi, track_phi, track_valid):
    track_phi_rel = kinematics.delta_phi(track_phi, jet_phi)
    track_phi_rel = ak.where(track_valid, track_phi_rel, 0)

    return track_phi_rel

def eta_rel(jet_eta, track_eta, track_valid):
    track_eta_rel = ak.where(track_valid, jet_eta-track_eta, 0)

    return track_eta_rel

def dr(phi_rel, eta_rel):
    return np.sqrt(np.power(phi_rel, 2) + np.power(eta_rel, 2))

def signed_2d_ip(d_0, sigma_d_0, eta_rel, track_valid):
    sign = np.sign(d_0 * np.sin(eta_rel))
    signed_ip = sign * np.abs(d_0 / sigma_d_0)
    signed_ip = ak.where(track_valid, signed_ip, 0)
    return signed_ip
    
