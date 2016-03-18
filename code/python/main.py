import matplotlib.pyplot as plt
from Polymer import Polymer

N = 500
T = 1
N_angles = 6


pol = Polymer(N, N_angles, T)
pol.populate()
plt.plot(pol.population[:,0], pol.population[:,1])
plt.show()
