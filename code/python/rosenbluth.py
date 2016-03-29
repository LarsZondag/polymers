import numpy as np
import matplotlib.pyplot as plt
from numba import jit

L = 250
N_angles = 6
sigma = 0.8
sigma2 = 0.8 ** 2
epsilon = 0.25
T = 1
pop_per_angle = 10
static_angles = np.linspace(0, 2 * np.pi, N_angles, False)
pos_new = np.zeros((N_angles, 2))
max_pop = 50 * pop_per_angle
ete = np.zeros((L, max_pop))
pol_weights = np.zeros((L, max_pop))
pol_weights[1, 0] = 1

mean_ete = np.zeros(L)

polymer_index = np.zeros(L, dtype=int)

initial_population = np.zeros((L, 2))
initial_population[1] = [1, 0]

@jit
def calc_energy(population, possible_positions, bead):
    interaction_energy = np.zeros(N_angles)
    for j in range(N_angles):
        for i in range(bead + 1):
            dx = possible_positions[j, 0] - population[i,0]
            dy = possible_positions[j, 1] - population[i,1]
            d2 = dx*dx + dy*dy
            ir2 = sigma2 / d2
            ir6 = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            interaction_energy[j] += 4 * epsilon * (ir12 - ir6)
    return interaction_energy

def calc_angles():
    random_angle = 2 * np.pi * np.random.rand()
    return static_angles + random_angle

def choose_angle(weights):
    cumsum = np.cumsum(weights)
    random = np.random.rand() * cumsum[-1]
    pos_index = np.digitize(random, cumsum)
    # if pos_index == N_angles:
    #     pos_index = np.random.randint(0, N_angles)
    return pos_index

def get_ete(population, bead):
        return np.linalg.norm(population[0] - population[bead]) ** 2

def add_bead(population, bead = 1, pol_weight = 1, perm = True):
    # print("Now at bead: ", bead + 1)
    if bead + 1 >= L or polymer_index[bead + 1] >= max_pop:
        return

    angles = calc_angles()
    pos_new[:, 0] = population[bead, 0] + np.cos(angles)
    pos_new[:, 1] = population[bead, 1] + np.sin(angles)

    energies = calc_energy(population, pos_new, bead)
    weights = np.exp(-energies/T)
    weights_sum = np.sum(weights)
    pol_weight *= weights_sum / (0.75 * N_angles)

    angle_index = choose_angle(weights)
    if angle_index == N_angles:
        return
    population[bead + 1] = pos_new[angle_index]

    ete[bead +1, polymer_index[bead + 1]] = get_ete(population, bead + 1)
    pol_weights[bead + 1, polymer_index[bead + 1]] = pol_weight

    pol_weight_3 = np.sum(pol_weights[2, :polymer_index[2]])
    pol_weights_mean = np.sum(pol_weights[bead + 1, :polymer_index[bead + 1]])
    polymer_index[bead + 1] +=1

    if perm:
        if pol_weight_3 != 0 and pol_weight != 0:
            limit_upper = 2 * pol_weights_mean.mean() / pol_weight_3
            limit_lower = 1.2 * pol_weights_mean.mean() / pol_weight_3
            if pol_weight > limit_upper:
                print("ENRICHING")
                add_bead(population, bead + 1, pol_weight / 2)
                add_bead(population, bead + 1, pol_weight / 2)
            elif pol_weight < limit_lower:
                if np.random.rand() < 0.5:
                    print("GOING TO PRUNE")
                    add_bead(population, bead + 1, 2 * pol_weight)
            else:
                # print("no perm")
                add_bead(population, bead + 1, pol_weight)
    else:
        add_bead(population, bead + 1, pol_weight, perm)


for i in range(pop_per_angle):
    add_bead(initial_population)


ete_avg = np.average(ete[2:], weights=pol_weights[2:], axis=1)
ete_var = np.average(ete[2:]**2, weights=pol_weights[2:], axis=1) - ete_avg**2
ete_error = np.sqrt(ete_var/pop_per_angle)

lengths = np.arange(2, L)

plt.loglog(lengths, ete_avg)

plt.xlim(2,L*1.3)
# plt.ylim(ete_avg[0],ete_avg[L-3]*1.3)

plt.show()
