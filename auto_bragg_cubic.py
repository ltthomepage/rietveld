from collections import defaultdict
import numpy as np
from typing import List, Tuple


def cal_d_via_unitcell(h:int, k:int, l:int, a: float) -> float:
    '''
    Calculate d spacing from lattice cell(a, b, c,α, β, γ).
    '''
    
    d_hkl = a / (h ** 2 + k ** 2 + l ** 2) ** 0.5
    return d_hkl


def cal_2theta_via_d(d_hkl: float, wavelength: float = 0.154, n: int = 1) ->float:
    '''
    Calculate Bragg angle from d spacing.
    '''
    sin_theta = n * wavelength / (2 * d_hkl)
    if 0 < sin_theta < 1:
        theta = np.degrees(np.arcsin(sin_theta))
        return 2 * theta
    return None

def cal_bragg_angles(a: float,wavelength: float = 0.154, n = 1,angle_min:float = 10, angle_max:float = 100) -> List[Tuple[float, float, List[Tuple[int, int, int]], int]]:

    bragg_data = defaultdict(list)
    for h in range(-9, 10):
        for k in range(-9, 10):
            for l in range(-9, 10):
                if (h,k,l) != (0,0,0):
                    d_hkl = cal_d_via_unitcell(h,k,l,a)
                    two_theta = cal_2theta_via_d(d_hkl)
                    if two_theta is not None and angle_min < two_theta < angle_max:
                        bragg_data[d_hkl].append((h,k,l))

    sorted_bragg_data = [(d_hkl, cal_2theta_via_d(d_hkl), planes, len(planes))
                         for d_hkl, planes in sorted(bragg_data.items(), key=lambda x: cal_2theta_via_d(x[0]))]
    
    return sorted_bragg_data



plane_list_fe = cal_bragg_angles(0.28)
for d, twotheta, planes, mul in plane_list_fe:
    print(np.round(d, 3), np.round(twotheta, 3), planes[0], mul)







