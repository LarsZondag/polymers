import numpy as np
from numba import jit
import matplotlib.pyplot as plt

N = 100
L=2

eps = 0.25
sigma = 0.8
sigma2 = sigma**2
T = 1

polymer = np.zeros((N, 2))
N_angles = 6
d_theta = 2 * np.pi / (N_angles + 1)
angles = np.linspace(d_theta - np.pi, N_angles * d_theta - np.pi, N_angles)
pol_weight = 1

# Initialize:
polymer[1,1] = 1

def calc_boltz_weights(bead):
    dr = polymer[bead-1] - polymer[bead-2]
    angle_offset = np.arctan2(dr[1],dr[0])
    # print("Angle offset: " + repr(angle_offset * 180 / np.pi))
    possible_positions = np.zeros((N_angles, 2))
    possible_positions[:,0] = polymer[bead-1, 0] + np.cos(angles + angle_offset)
    possible_positions[:,1] = polymer[bead-1, 1] + np.sin(angles + angle_offset)
    energies = calc_energy(possible_positions, bead)
    weights = np.exp(-energies/T)
    W = np.sum(weights)
    weights = np.cumsum(weights/W)
    # print(weights)
    return possible_positions, weights, W


def calc_energy(possible_positions, bead):
    interaction_energy = np.zeros(N_angles)
    interaction_energy += 4*eps*(sigma**12-sigma**6)
    for j in range(N_angles):
        for i in range(bead-2):
            possible_positions_dx = possible_positions[j, 0] - polymer[i,0]
            possible_positions_dy = possible_positions[j, 1] - polymer[i,1]
            distance = possible_positions_dx*possible_positions_dx + possible_positions_dy*possible_positions_dy
            ir2 = sigma2 / distance
            ir6 = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            interaction_energy[j] += 4 * eps * (ir12 - ir6)
    print(interaction_energy)
    return interaction_energy



def add_bead(polymer, pol_weight, L):
    possible_positions, w_i, W = calc_boltz_weights(L)
    new_position = possible_positions[int (np.random.uniform() * N_angles)]
    # index = [ n for n, i in enumerate(w_i) if i>np.random.uniform()][0]
    # new_position = possible_positions[index]
    polymer[L] = new_position
    if L < N - 1:
        add_bead(polymer, pol_weight, L+1)
    else:
        return polymer


add_bead(polymer, pol_weight, L)

plt.plot(polymer[:,0], polymer[:,1])
plt.show()
