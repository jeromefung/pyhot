'''
test_pyhot.py

Test suite for use with pytest.

Author: Jerome Fung (jfung@ithaca.edu)
'''

import numpy as np

from pyhot import SLM
import pytest

@pytest.fixture
def my_slm():
    return SLM(nx = 300, ny = 480, px = 8., wavelen = 1.064, f = 3333.33)

@pytest.fixture
def trap_points():
    return np.array([[2, 3, 0],
                     [-4, 1, 0],
                     [1, -2, 2]])

def test_holo(my_slm, trap_points):
    meths = ['spl']
    for meth in meths:
        holo = my_slm.calc_holo(trap_points, method = meth)

    
