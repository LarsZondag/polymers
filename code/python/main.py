import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import bisect

N = 250
L=2

eps = 0.25
sigma = 0.8
sigma2 = sigma**2
T = 1

polymer = np.zeros((N, 2))
N_angles = 6
d_theta = 2 * np.pi / (N_angles)
static_angles = np.linspace(0, 2*np.pi-d_theta, N_angles)
pol_weight = 1
random_numbers = np.random.uniform(0, 1, N)

# Initialize:
polymer[1,1] = 1


def calc_boltz_weights(bead):
    possible_positions = np.zeros((N_angles, 2))
    random_angle = 2*np.pi*np.random.rand()
    angles = static_angles + random_angle
    # print(angles*180/np.pi)
    possible_positions[:,0] = polymer[bead-1, 0] + np.cos(angles)
    possible_positions[:,1] = polymer[bead-1, 1] + np.sin(angles)
    energies = calc_energy(possible_positions, bead)
    # print(energies)
    weights = np.exp(-energies/T)
    W = np.sum(weights)
    weights = np.cumsum(weights)
    # print(weights)
    return possible_positions, weights, W


def calc_energy(possible_positions, bead):
    interaction_energy = np.zeros(N_angles)
    # interaction_energy += 4*eps*(sigma**12-sigma**6)
    for j in range(N_angles):
        for i in range(bead):
            dx = possible_positions[j, 0] - polymer[i,0]
            dy = possible_positions[j, 1] - polymer[i,1]
            d2 = dx*dx + dy*dy
            ir2 = sigma2 / d2
            ir6 = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            interaction_energy[j] += 4 * eps * (ir12 - ir6)
    return interaction_energy

for i in range(L, N):
    #print("bead #: ", i)
    possible_positions, w_i, W = calc_boltz_weights(i)
    score = W * random_numbers[i]
    index = bisect.bisect(w_i, score)
    index = min(index, N_angles-1)
    #print("Chosen chance: ", w_i[index]/W)
    polymer[i,:] = possible_positions[index]





# plt.plot(polymer[:,0], polymer[:,1], linestyle='-.', marker='o')
plt.plot(polymer[:,0], polymer[:,1])
plt.show()
