import numpy as np
import awkward as ak

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

def rel_ang(track, jet):
    return track - jet

def frac_pt(track, jet):
    return track / jet

def dr(rel_phi, rel_eta):
    return np.sqrt(np.power(rel_phi, 2) + np.power(rel_eta, 2))
