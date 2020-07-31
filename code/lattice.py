import numpy as np
import copy


class Lattice:
    def __init__(self, N, k, l, eps):
        self.N = N
        self.k = k
        self.l = l
        self.eps = eps        
        self.sqrt_eps = np.sqrt(2 * self.eps)
        
        self.phi = np.random.randn(N, N)
        self.action = self.get_action()
    
    def get_action(self):
        return np.sum((-2. * self.k * np.multiply(
            self.phi,
            np.roll(self.phi, 1, 0)
            + np.roll(self.phi, 1, 1))
            + (1 - 2 * self.l) * self.phi**2
            + self.l * self.phi**4))

    def get_drift(self):
        return (2. * self.k * 
                (np.roll(self.phi, 1, 0) +
                 np.roll(self.phi, -1, 0) +
                 np.roll(self.phi, 1, 1) +
                 np.roll(self.phi, -1, 1)) +
                2 * self.phi * (2 * self.l *
                (1 - self.phi**2) - 1))

    def get_hamiltonian(self, chi, action):
        return 0.5 * np.sum(chi**2) + action

    def langevin(self):
        chi = np.random.randn(self.N, self.N)

        self.phi += (self.eps * self.get_drift() +
                     self.sqrt_eps * chi)

    def hmc(self, n_steps=100):
        phi_0 = copy.deepcopy(self.phi)
        chi = np.random.randn(self.N, self.N)

        S_0 = self.action
        H_0 = self.get_hamiltonian(chi, S_0)

        chi += 0.5 * self.eps * self.get_drift()

        for i in range(n_steps):
            self.phi += self.eps * chi

            if i == n_steps-1:
                chi += 0.5 * self.eps * self.get_drift()
            else:
                chi += self.eps * self.get_drift()

        self.action = self.get_action()
        dH = self.get_hamiltonian(chi, self.action) - H_0

        if dH > 0:
            if np.random.rand() >= np.exp(-dH):
                self.phi = phi_0
                self.action = S_0
