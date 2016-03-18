import matplotlib.pyplot as plt
from Polymer import Polymer
from mpl_toolkits.mplot3d import Axes3D

N = 700
T = 1
N_angles = 4


pol = Polymer(N, N_angles, T)
pol.populate()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pol.population[:,0], pol.population[:,1], pol.population[:,2])
fig.show()


