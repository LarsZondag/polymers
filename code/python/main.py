import numpy as np
from numba import jit

N = 100
L=0

eps = 0.25
sigma = 0.8
T = 1

locations = np.array((N, 2))
N_angles = 6
d_theta = np.pi / (N_angles + 1)
angles = np.linspace(d_theta, N_angles * d_theta, N_angles)

def calc_boltz_weights():

    pass

def add_bead(polymer, pol_weight, L):

    pass