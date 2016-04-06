import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import curve_fit

L = 250
N_angles = 6
sigma = 0.8
sigma2 = 0.8 ** 2
epsilon = 0.25
T = 1
pop_per_angle = 250
static_angles = np.linspace(0, 2 * np.pi, N_angles, False)
pos_new = np.zeros((N_angles, 2))
max_pop = 50 * pop_per_angle
ete = np.zeros((L, max_pop))
R_G_2 = np.zeros((L, max_pop))
pol_weights = np.zeros((L, max_pop))
pol_weights[1, 0] = 1
prunes = np.zeros((L))
enrichments = np.zeros((L))

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

def calc_radius_of_gyration(population, bead):
    R_G = np.mean(population[:bead], axis=0)
    R_G_x_2 = np.power(population[:bead, 0] - R_G[0], 2)
    R_G_y_2 = np.power(population[:bead, 1] - R_G[1], 2)
    R_G_2 = np.mean(R_G_x_2 + R_G_y_2)
    return R_G_2

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
    R_G_2[bead + 1, polymer_index[bead + 1]] = calc_radius_of_gyration(population, bead + 1)
    pol_weights[bead + 1, polymer_index[bead + 1]] = pol_weight

    pol_weight_3 = np.sum(pol_weights[2, :polymer_index[2]])
    pol_weights_mean = np.sum(pol_weights[bead + 1, :polymer_index[bead + 1]])
    polymer_index[bead + 1] +=1

    if perm:
        if pol_weight_3 != 0 and pol_weight != 0:
            limit_upper = 2 * pol_weights_mean.mean() / pol_weight_3
            limit_lower = 1.2 * pol_weights_mean.mean() / pol_weight_3
            if pol_weight > limit_upper:
                # print("ENRICHING")
                enrichments[bead] += 1
                add_bead(population, bead + 1, pol_weight / 2)
                add_bead(population, bead + 1, pol_weight / 2)
            elif pol_weight < limit_lower:
                if np.random.rand() < 0.5:
                    # print("GOING TO PRUNE")
                    prunes[bead] += 1
                    add_bead(population, bead + 1, 2 * pol_weight)
            else:
                # print("no perm")
                add_bead(population, bead + 1, pol_weight)
    else:
        add_bead(population, bead + 1, pol_weight, perm)


for i in range(pop_per_angle):
    print("Starting polymer ", i, " out of ", pop_per_angle)
    add_bead(initial_population)


# Statistics

ete_avg = np.average(ete[2:], weights=pol_weights[2:], axis=1)
ete_var = np.average(ete[2:]**2, weights=pol_weights[2:], axis=1) - ete_avg**2
ete_error = np.sqrt(ete_var/pop_per_angle)

R_G_2_avg = np.average(R_G_2[2:], weights=pol_weights[2:], axis=1)
R_G_2_var = np.average(R_G_2[2:]**2, weights=pol_weights[2:], axis=1) - R_G_2_avg**2
R_G_2_error = np.sqrt(R_G_2_var/polymer_index[2:])

lengths = np.arange(2, L)

# Defining functions

def func(x, a, b):
    return a * np.power(x - 1, b)

# Fitting the data

popt, pcov = curve_fit(func, lengths, ete_avg, bounds=(0, [3., 3.]))
print(popt)

# Plotting

ax = plt.subplot(211)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')

plt.errorbar(lengths, ete_avg, yerr=ete_error, linestyle='None', marker='x')
plt.plot(lengths, func(lengths, popt[0], popt[1]))
plt.plot(lengths, polymer_index[2:], linestyle='None', marker='o', mfc='None')

ax = plt.subplot(212)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
plt.plot(lengths, R_G_2_avg)
plt.show()
