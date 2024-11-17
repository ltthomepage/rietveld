import numpy as np


def cal_d_via_a(hkl_list, lattice_parameter=1):
    """
    Calculate the interplanar spacings (d_hkl) for a list of Miller indices.
    Parameters:
    - hkl_list
    - lattice_parameter
    Returns:
    - d_list: List of interplanar spacings for each Miller index.
    """
    # Initialize a list to store the interplanar spacings d_hkl
    d_list = [lattice_parameter / np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2) for hkl in hkl_list]
    return d_list

def cal_2theta_via_d(d_list, wavelength=0.154):
    """
    Calculate the 2*theta angles from a list of interplanar spacings (d_hkl).
    Parameters:
    - d_list: List or array 
    - wavelength: X-ray wavelength 
    Returns:
    - theta2_list
    """
    # Initialize a list to store the 2*theta angles
    theta2_list = [np.arcsin(wavelength / 2 / d) * 2 * 180 / np.pi for d in d_list]
    return theta2_list


