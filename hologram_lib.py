import numpy as np
from numba import jit
from numba.types import float64, uint16, int32


@jit(uint16[:,:](float64[:,:], float64[:,:], float64[:,:], float64, float64),
     nopython=True)
def calc_holo_spl(pts, xs, ys, wavelen, f):
    phase_sum = np.empty_like(xs, dtype = np.complex64)
    
    for i in range(pts.shape[0]):
        Delta_m_lens = np.pi * pts[i, 2] / (wavelen * f**2) * (xs**2 + ys**2)
        Delta_m_grating = 2 * np.pi / (wavelen * f) * (xs * pts[i, 0] +
                                                       ys * pts[i, 1])
        Delta_m = Delta_m_lens + Delta_m_grating
        phase_sum += np.exp(1j * Delta_m)
    
    # np.angle between -pi and pi, shift to 0 to 2 pi 
    holo = np.angle(phase_sum) + np.pi

    # scale to 8 bit
    holo *= (255 / (2 * np.pi))

    # round
    return np.round(holo, 0, np.empty_like(holo)).astype(np.uint16)

    
    
