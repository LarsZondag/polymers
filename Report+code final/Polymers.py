import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import curve_fit

# L = 250 #Length of beads
# number_of_angles = 6
# sigma = 0.8
# sigma2 = 0.8 ** 2
# epsilon = 0.25
# T = 1
# pop_polymers = 10000
# global_static_angles = np.linspace(0, 2 * np.pi, number_of_angles, False)
# max_pop = 50 * pop_polymers # Maximum population, this is done in case there are more enrichments than prunes
# ete = np.zeros((L, max_pop)) # End-to-end distance squared
# R_G_2 = np.zeros((L, max_pop)) # Gyration radius squared
# # Initialize the starting polymer weights. This is the same for every starting polymer:
# pol_weights = np.zeros((L, max_pop))
# pol_weights[1, 0] = 1
# pol_weights[0, 0] = 1
#
# prunes = np.zeros((L))
# enrichments = np.zeros((L))
#
# mean_ete = np.zeros(L)
#
# polymer_index = np.zeros(L, dtype=int)
#
# initial_population = np.zeros((L, 2))
# initial_population[1] = [1, 0]
#
# @jit
# def calc_energy(population, possible_positions, bead):
#     loop_size = np.size(possible_positions[:, 0])
#     interaction_energy = np.zeros(loop_size)
#     for j in range(loop_size):
#         for i in range(bead + 1):
#             dx = possible_positions[j, 0] - population[i,0]
#             dy = possible_positions[j, 1] - population[i,1]
#             d2 = dx*dx + dy*dy
#             ir2 = sigma2 / d2
#             ir6 = ir2 * ir2 * ir2
#             ir12 = ir6 * ir6
#             interaction_energy[j] += 4 * epsilon * (ir12 - ir6)
#     return interaction_energy
#
# @jit
# def calc_radius_of_gyration(population, bead):
#     R_G = np.mean(population[:bead], axis=0)
#     R_G_x_2 = np.power(population[:bead, 0] - R_G[0], 2)
#     R_G_y_2 = np.power(population[:bead, 1] - R_G[1], 2)
#     R_G_2 = np.mean(R_G_x_2 + R_G_y_2)
#     return R_G_2
#
# def choose_angle(weights):
#     cumsum = np.cumsum(weights)
#     random = np.random.rand() * cumsum[-1]
#     return np.digitize(random, cumsum)
#
# def add_bead(population, static_angles, bead = 1, pol_weight = 1, perm = True):
#     # Make sure the addition of the polymer will not exceed the maximum population size
#     if polymer_index[bead + 1] >= max_pop:
#         print("The population got too big")
#         return
#
#     # Determine the angles to the new position and initiate the new positions
#     N_angles = np.size(static_angles)
#     angles = 2 * np.pi * np.random.rand() + static_angles
#     pos_new = np.vstack((population[bead, 0] + np.cos(angles), population[bead, 1] + np.sin(angles))).T
#
#     # Determine the energies corresponding to the angles an calculate their weights
#     energies = calc_energy(population, pos_new, bead)
#     weights = np.exp(-energies/T)
#     weights_sum = np.sum(weights)
#     pol_weight *= weights_sum / (0.75 * N_angles)
#
#     # Choose an angle based on their weights
#     angle_index = choose_angle(weights)
#
#     # If the weights were so low the roulette algorithm was unable to select an appropriate angle,
#     # the polymer will be discarded
#     if angle_index == N_angles:
#         return
#     population[bead + 1] = pos_new[angle_index]
#
#     # Perform end-to-end and gyration radius measurements and add them to the array.
#     # Also update the polymer weights array
#     ete[bead +1, polymer_index[bead + 1]] = np.linalg.norm(population[0] - population[bead + 1]) ** 2
#     R_G_2[bead + 1, polymer_index[bead + 1]] = calc_radius_of_gyration(population, bead + 1)
#     pol_weights[bead + 1, polymer_index[bead + 1]] = pol_weight
#     pol_weight_3 = np.sum(pol_weights[2, :polymer_index[2]])
#     pol_weights_mean = np.sum(pol_weights[bead + 1, :polymer_index[bead + 1]])
#
#     polymer_index[bead + 1] +=1
#
#     # Check to see if the next bead can still be grown or if the polymer has reached max length
#     if bead + 2 >= L:
#         return
#     elif perm:
#         if pol_weight_3 != 0 and pol_weight != 0:
#             limit_upper = 2 * pol_weights_mean.mean() / pol_weight_3
#             limit_lower = 1.2 * pol_weights_mean.mean() / pol_weight_3
#             if pol_weight > limit_upper:
#                 # print("ENRICHING")
#                 enrichments[bead] += 1
#                 add_bead(population, static_angles, bead + 1, pol_weight / 2)
#                 add_bead(population, static_angles, bead + 1, pol_weight / 2)
#             elif pol_weight < limit_lower:
#                 if np.random.rand() < 0.5:
#                     # print("GOING TO PRUNE")
#                     prunes[bead] += 1
#                     add_bead(population, static_angles, bead + 1, 2 * pol_weight)
#             else:
#                 # print("no perm")
#                 add_bead(population, static_angles, bead + 1, pol_weight)
#     else:
#         add_bead(population, static_angles, bead + 1, pol_weight, perm)
#
#
# for i in range(pop_polymers):
#     # print("Starting polymer ", i, " out of ", pop_polymers)
#     add_bead(initial_population, global_static_angles)
#
#
# # Statistics: averaging the data, determining the error and fitting the data
#
# ete_avg = np.average(ete[2:], weights=pol_weights[2:], axis=1)
# ete_var = np.average(ete[2:]**2, weights=pol_weights[2:], axis=1) - ete_avg**2
# ete_error = np.sqrt(ete_var/pop_polymers)
#
# R_G_2_avg = np.average(R_G_2[2:], weights=pol_weights[2:], axis=1)
# R_G_2_var = np.average(R_G_2[2:]**2, weights=pol_weights[2:], axis=1) - R_G_2_avg**2
# R_G_2_error = np.sqrt(R_G_2_var/polymer_index[2:])
#
# lengths = np.arange(2, L)
#
# # Defining functions
#
def func(x, a, b):
    return a * np.power(x , b)
#
# # Fitting the data
#
# ete_params, ete_pcov = curve_fit(func, lengths, ete_avg, bounds=(0, [10., 3.]))
# print("Coefficients for mean square end-to-end distance: ", ete_params)
#
# R_G_2_params, R_G_2_pcov = curve_fit(func, lengths, R_G_2_avg, bounds=(0, [10., 3.]))
# print("Coefficients for mean square gyration radius: ", R_G_2_params)

# Plotting
# End to end distance plotting
ete_fig = plt.figure()
ax = plt.subplot(111)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.set_ylim([1, pop_polymers + 0.1*pop_polymers])

plt_ete = ax.errorbar(lengths, ete_avg, yerr=ete_error, linestyle='None', marker='x')
plt_ete_fit, = ax.plot(lengths, func(lengths, ete_params[0], ete_params[1]))
plt_ete_pop, = ax.plot(lengths, polymer_index[2:], linestyle='None', marker='o', mfc='None')
ax.legend([plt_ete, plt_ete_fit, plt_ete_pop],["$<R^2_{end}>$", "$y = %4.2f \cdot (N-1)^{%4.2f}$" % (ete_params[0], ete_params[1]), "$N_{pop}$"],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.xlabel('polymer size $N$')
plt.ylabel('mean square of end-to-end distance $<R^2_{end}>$')
plt.show()

R_G_fig = plt.figure()
ax = plt.subplot(111)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')

plt_R_G_2 = ax.errorbar(lengths, R_G_2_avg, yerr = R_G_2_error, linestyle='None', marker='x')
plt_R_G_2_fit, = ax.plot(lengths, func(lengths, R_G_2_params[0], R_G_2_params[1]))
ax.legend([plt_R_G_2, plt_R_G_2_fit], ["$<R^2_{g}>$", "$y = %4.2f \cdot (N-1)^{%4.2f}$" % (R_G_2_params[0], R_G_2_params[1])],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.xlabel('polymer size $N$')
plt.ylabel('mean square gyration radius $<R^2_{g}>$')
plt.show()
