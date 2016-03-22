import numpy as np
import matplotlib.pyplot as plt

L = 15
N_angles = 6
sigma = 0.8
sigma2 = 0.8 ** 2
epsilon = 0.25
T = 1
pop_per_angle = 100
static_angles = np.linspace(0, 2 * np.pi, N_angles, False)
pos_new = np.zeros((N_angles, 2))
max_pop = 5 * pop_per_angle
end_to_end = np.zeros((L, max_pop))
pol_weights = np.zeros((L, max_pop))


mean_end_to_end = np.zeros(L)

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

def add_bead(population, bead = 0, pol_weight = 1, perm = True):
    # print("Now at bead: ", bead + 1)
    angles = calc_angles()
    pos_new[:, 0] = population[bead, 0] + np.cos(angles)
    pos_new[:, 1] = population[bead, 1] + np.sin(angles)

    energies = calc_energy(population, pos_new, bead)
    weights = np.exp(-energies/T)
    weights_sum = np.sum(weights)
    pol_weight *= weights_sum / (0.75 * N_angles)

    angle_index = choose_angle(weights)
    population[bead + 1] = pos_new[angle_index]
    print("Added bead: ", bead + 1)
    # print("Weight of this chain is: ", pol_weight)

    end_to_end[bead +1, polymer_index[bead + 1]] = get_end_to_end(population, bead + 1)
    pol_weights[bead + 1, polymer_index[bead + 1]] = pol_weight

    # print(pol_weights[2])
    polymer_index[bead + 1] +=1
    pol_weight_3 = pol_weights[2, polymer_index[2] - 1]
    # print("pol_weight_3: ", pol_weight_3)
    pol_weights_mean = np.mean(pol_weights[bead + 1, :polymer_index[bead + 1]])
    # print(pol_weights_mean)



    limit_upper = 2.0 * pol_weight_3 / pol_weights_mean
    limit_lower = 1.2 * pol_weight_3 / pol_weights_mean

    if perm:
        if polymer_index[bead+1] > max_pop - 2 and bead < L - 2:
            add_bead(population, bead + 1, pol_weight)
        elif bead < L - 2:
            if pol_weight > limit_upper:
                print("ENRICHING")
                add_bead(population, bead + 1, pol_weight / 2)
                add_bead(population, bead + 1, pol_weight / 2)
            elif pol_weight < limit_lower:
                print("GOING TO PRUNE")
                if np.random.rand(1) < 0.5:
                    add_bead(population, bead + 1, 2 * pol_weight)
            else:
                print("no perm")
                add_bead(population, bead + 1, pol_weight)
        else:
            return
    else:
        if bead < L - 1:
            add_bead(population, bead + 1, pol_weight, perm)


for i in range(pop_per_angle):
    print("Going to add polymer: ", i)
    add_bead(initial_population)

end_to_end_avg = np.average(end_to_end[2:], weights=pol_weights[2:], axis=1)
end_to_end_var = np.average(end_to_end[2:]**2, weights=pol_weights[2:], axis=1) - end_to_end_avg**2
end_to_end_error = np.sqrt(end_to_end_var/pop_per_angle)

lengths = np.arange(2, L)

plt.loglog(lengths, end_to_end_avg)

plt.xlim(2,L*1.3)
plt.ylim(end_to_end_avg[0],end_to_end_avg[L-3]*1.3)

plt.show()
