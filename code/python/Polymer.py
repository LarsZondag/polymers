import numpy as np
from operator import itemgetter
from numba import jit
import bisect


class Polymer:
    def __init__(self, length, number_of_angles, temperature):
        self.L = length
        self.N_angles = number_of_angles
        self.N_angles_total = self.N_angles ** 2
        self.population = np.zeros((self.L, 3))
        self.azimuthal_angles = np.linspace(0, 2*np.pi, self.N_angles, False)
        self.polar_angles = np.linspace(0, np.pi, self.N_angles)
        self.bead = 1
        self.sigma = 0.8
        self.sigma2 = self.sigma ** 2
        self.epsilon = 0.25
        self.T = temperature


    def populate(self):
        self.random_azimuthal_angles = 2 * np.pi * np.random.rand(self.L)
        self.random_polar_angles = np.pi * np.random.rand(self.L)
        while self.bead < self.L:
            new_pos = np.zeros((self.N_angles_total, 3))
            polar_angles, azimuthal_angles = self.calc_angles()
            for i in range(self.N_angles):
                for j in range(self.N_angles):
                    new_pos[i*self.N_angles + j,0] = self.population[self.bead - 1, 0] + np.sin(polar_angles[i]) * np.cos(azimuthal_angles[j])
                    new_pos[i*self.N_angles + j,1] = self.population[self.bead - 1, 1] + np.sin(polar_angles[i]) * np.sin(azimuthal_angles[j])
                    new_pos[i*self.N_angles + j,2] = self.population[self.bead - 1, 2] + np.cos(polar_angles[i])
            energy = self.calc_energy(new_pos)
            weights = self.calc_weights(energy)
            if not np.any(weights):
                print("I GOT STUCK! I left at bead: ",self.bead, "I will try again.")
                self.bead = 1
                self.population = np.zeros((self.L, 3))
                self.populate()
                break
            index = self.choose_angle(weights)
            self.population[self.bead] = new_pos[index]
            self.bead += 1
        return self.population

    def calc_energy(self, possible_positions):
        interaction_energy = np.zeros(self.N_angles_total)
        for j in range(self.N_angles):
            for i in range(self.bead):
                dx = possible_positions[j, 0] - self.population[i,0]
                dy = possible_positions[j, 1] - self.population[i,1]
                dz = possible_positions[j, 2] - self.population[i,2]
                d2 = dx*dx + dy*dy + dz * dz
                ir2 = self.sigma2 / d2
                ir6 = ir2 * ir2 * ir2
                ir12 = ir6 * ir6
                interaction_energy[j] += 4 * self.epsilon * (ir12 - ir6)
        return interaction_energy

    def calc_angles(self):
        random_azimuthal_angle = self.random_azimuthal_angles[self.bead]
        random_polar_angle = self.random_polar_angles[self.bead]
        azimuthal_angle = self.azimuthal_angles + random_azimuthal_angle
        polar_angle = (self.polar_angles + random_polar_angle) % np.pi
        return polar_angle, azimuthal_angle

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


