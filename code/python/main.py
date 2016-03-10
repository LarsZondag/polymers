import numpy as np
from numba import jit

N = 100
L=0

eps = 0.25
sigma = 0.8
T = 1

polymer = np.zeros((N, 2))
N_angles = 6
d_theta = np.pi / (N_angles + 1)
angles = np.linspace(d_theta, N_angles * d_theta, N_angles)

# Initialize:
polymer[1,0] = 1

def calc_boltz_weights(L):
    dr = polymer[L-1] - polymer[L-2]
    if dr[2] == 0:
        angle_offset = 0
    else:
        angle_offset = np.arctan(dr[1]/dr[2])
    possible_positions = polymer[L-1, 0] + np.cos(angles + angle_offset)


def calc_energy(possible_positions):
    pass

def add_bead(polymer, pol_weight, L):

    pass