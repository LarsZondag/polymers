import numpy as np
import matplotlib.pyplot as plt

L = 25
N_angles = 6
sigma = 0.8
sigma2 = 0.8 ** 2
epsilon = 0.25
T = 1
pop_per_angle = 10
static_angles = np.linspace(0, 2 * np.pi, N_angles, False)
pos_new = np.zeros((N_angles, 2))
end_to_end = np.zeros((L, 5 * pop_per_angle))
pol_weights = np.zeros((L, 5 * pop_per_angle))

polymer_index = np.zeros(L, dtype=int)

initial_population = np.zeros((L, 2))

def calc_energy(population, possible_positions, bead):
    interaction_energy = np.zeros(N_angles)
    for j in range(N_angles):
        for i in range(bead):
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
    if not np.any(weights):
        return np.random.randint(0, N_angles)
    else:
        cumsum = np.cumsum(weights)
        random = np.random.uniform(0, cumsum[-1])
        pos_index = 0
        while random >= cumsum[pos_index] and pos_index <= N_angles:
            pos_index += 1
        return pos_index

def get_end_to_end(population, bead):
        return np.linalg.norm(population[0] - population[bead]) ** 2

def add_bead(population, bead = 1, pol_weight = 1, perm = True):
    angles = calc_angles()
    pos_new[:, 0] = population[bead-1, 0] + np.cos(angles)
    pos_new[:, 1] = population[bead-1, 1] + np.sin(angles)

    energies = calc_energy(population, pos_new, bead)
    weights = np.exp(-energies/T)
    weights_sum = np.sum(weights)
    pol_weight *= weights_sum / (0.75 * N_angles)

    angle_index = choose_angle(weights)
    population[bead] = pos_new[angle_index]

    end_to_end[bead, polymer_index[bead]] = get_end_to_end(population, bead)
    pol_weights[bead, polymer_index[bead]] = pol_weight

    polymer_index[bead] +=1


    if bead + 1 < L:
        add_bead(population, bead + 1, pol_weight)
    else:
        return

for i in range(pop_per_angle):
    print("Going to add polymer: ", i)
    add_bead(initial_population)
