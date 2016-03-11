import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import bisect

N = 100000
L=2

eps = 0.25
sigma = 0.8
sigma2 = sigma**2
T = 1

polymer = np.zeros((N, 2))
N_angles = 3
d_theta = 2 * np.pi / (N_angles + 1)
# angles = np.linspace(d_theta - np.pi, N_angles * d_theta - np.pi, N_angles)
pol_weight = 1
random_numbers = np.random.uniform(0, 1, N)

# Initialize:
polymer[1,1] = 1


def calc_boltz_weights(bead):
    possible_positions = np.zeros((N_angles, 2))
    angles = np.random.uniform(0, 2* np.pi, N_angles)
    possible_positions[:,0] = polymer[bead-1, 0] + np.cos(angles)
    possible_positions[:,1] = polymer[bead-1, 1] + np.sin(angles)
    energies = calc_energy(possible_positions, bead)
    weights = np.exp(-energies/T)
    W = np.sum(weights)
    weights = np.cumsum(weights)/W
    return possible_positions, weights, W

@jit
def calc_energy(possible_positions, bead):
    size = possible_positions[:,0].size
    interaction_energy = np.zeros(size)
    interaction_energy += 4*eps*(sigma**12-sigma**6)
    for j in range(size):
        for i in range(bead-1):
            possible_positions_dx = possible_positions[j, 0] - polymer[i,0]
            possible_positions_dy = possible_positions[j, 1] - polymer[i,1]
            distance = possible_positions_dx*possible_positions_dx + possible_positions_dy*possible_positions_dy
            ir2 = sigma2 / distance
            ir6 = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            interaction_energy[j] += 4 * eps * (ir12 - ir6)
    return interaction_energy

for i in range(L, N):
    possible_positions, w_i, W = calc_boltz_weights(i)
    index = bisect.bisect_left(w_i/random_numbers[i], 1)
    new_position = possible_positions[index]
    polymer[i] = new_position



plt.plot(polymer[:,0], polymer[:,1])
plt.show()
