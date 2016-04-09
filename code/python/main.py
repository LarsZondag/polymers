import matplotlib.pyplot as plt
from Polymer import Polymer
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import curve_fit

starting_length = 2
stop_length = 64
length_steps = 5
lengths = np.linspace(starting_length, stop_length, length_steps)
lengths = np.ceil(lengths)
T = 1
N_angles = 6
pop_size = 10000
N = 250


pol = Polymer(N, N_angles, T)
pol.populate()
print(pol.get_end_to_end())

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
plt.plot(pol.population[:,0], pol.population[:,1])
plt.show()

# end_to_end = np.zeros((length_steps, pop_size))
# for i in range(length_steps):
#     polymer = Polymer(lengths[i], N_angles, T)
#     for j in range(pop_size):
#         polymer.populate()
#         end_to_end[i, j] = polymer.get_end_to_end()
#
# mean_ETE = np.mean(end_to_end, axis= 1)
#
# def func (N, nu):
#     return np.power(N, nu)
#
# popt, pcov = curve_fit(func, lengths, mean_ETE)
# print("popt: ", popt)
# print("pcov: ", pcov)



# plt.loglog(lengths, mean_ETE, lengths, np.power(lengths, popt))
# plt.show()


