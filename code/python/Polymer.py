import numpy as np
from operator import itemgetter
from numba import jit
import bisect


class Polymer:
    def __init__(self, length, number_of_angles, temperature):
        self.L = length
        self.N_angles = number_of_angles
        self.population = np.zeros((self.L, 2))
        self.angles = np.linspace(0, 2*np.pi, self.N_angles, False)
        self.bead = 1
        self.sigma = 0.8
        self.sigma2 = self.sigma ** 2
        self.epsilon = 0.25
        self.T = temperature


    def populate(self):
        self.pol_weight = 1
        self.random_angles = 2 * np.pi * np.random.rand(self.L)
        print(self.random_angles)
        while self.bead < self.L:
            new_pos = np.zeros((self.N_angles, 2))
            angles = self.calc_angles()
            new_pos[:,0] = self.population[self.bead - 1, 0] + np.cos(angles)
            new_pos[:,1] = self.population[self.bead - 1, 1] + np.sin(angles)
            energy = self.calc_energy(new_pos)
            weights = self.calc_weights(energy)
            if not np.any(weights):
                print("I GOT STUCK! I left at bead: ",self.bead, "I will try again.")
                self.bead = 1
                self.population = np.zeros((self.L, 2))
                self.populate()
                break
            index = self.choose_angle(weights)
            self.population[self.bead] = new_pos[index]
            # self.pol_weight *= np.sum(weights)
            self.bead += 1
        # print("polweight: ", self.pol_weight)
        return self.population

    def calc_energy(self, possible_positions):
        interaction_energy = np.zeros(self.N_angles)
        for j in range(self.N_angles):
            for i in range(self.bead):
                dx = possible_positions[j, 0] - self.population[i,0]
                dy = possible_positions[j, 1] - self.population[i,1]
                d2 = dx*dx + dy*dy
                ir2 = self.sigma2 / d2
                ir6 = ir2 * ir2 * ir2
                ir12 = ir6 * ir6
                interaction_energy[j] += 4 * self.epsilon * (ir12 - ir6)
        return interaction_energy

    def calc_angles(self):
        random_angle = self.random_angles[self.bead]
        angles = self.angles + random_angle
        return angles

    def calc_weights(self, energy):
        weights = np.exp(-energy/self.T)
        return weights

    def choose_angle(self, weights):
        cumsum = np.cumsum(weights)
        random = np.random.uniform(0, cumsum[-1])
        pos_index = 0
        while random >= cumsum[pos_index] and pos_index <= self.N_angles:
            pos_index += 1
        return pos_index

    def get_end_to_end(self):
        return np.linalg.norm(self.population[0] - self.population[-1]) ** 2


