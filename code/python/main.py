import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import bisect

N = 4
L=2

eps = 0.25
sigma = 0.8
sigma2 = sigma**2
T = 1

polymer = np.zeros((N, 2))
N_angles = 4
d_theta = 2 * np.pi / (N_angles)
angles = np.linspace(0, 2*np.pi-d_theta, N_angles)
dr = np.array(([1, 0, -1, 0],[0, 1, 0, -1]))
pol_weight = 1
random_numbers = np.random.uniform(0, 1, N)

# Initialize:
polymer[1,1] = 1


def calc_boltz_weights(bead):
    possible_positions = np.zeros((N_angles, 2))
    possible_positions[:,0] = polymer[bead-1, 0] + dr[0,:]
    possible_positions[:,1] = polymer[bead-1, 1] + dr[1,:]
    plausible_positions, energies = calc_energy(possible_positions, bead)
    weights = np.exp(-energies/T)
    W = np.sum(weights)
    weights = np.cumsum(weights)/W
    return plausible_positions, weights, W


def calc_energy(possible_positions, bead):
    plausible_positions = np.copy(possible_positions)
    for i in range(N_angles):
        if plausible_positions[i,0] in polymer[:bead,0]:
            if plausible_positions[i,1] in polymer[:bead,1]:
                plausible_positions = np.delete(plausible_positions, i, 0)

    size = plausible_positions[:,0].size
    interaction_energy = np.zeros(size)
    interaction_energy += 4*eps*(sigma**12-sigma**6)
    for i in range(bead-1):
        for j in range(size):
            possible_positions_dx = plausible_positions[j, 0] - polymer[i,0]
            possible_positions_dy = plausible_positions[j, 1] - polymer[i,1]
            distance = possible_positions_dx*possible_positions_dx + possible_positions_dy*possible_positions_dy
            ir2 = sigma2 / distance
            ir6 = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            interaction_energy[j] += 4 * eps * (ir12 - ir6)
    print(plausible_positions)
    return plausible_positions, interaction_energy

for i in range(L, N):
    possible_positions, w_i, W = calc_boltz_weights(i)
    index = bisect.bisect_left(w_i/random_numbers[i], 1)
    new_position = possible_positions[index]
    polymer[i] = new_position



plt.plot(polymer[:,0], polymer[:,1])
plt.show()
