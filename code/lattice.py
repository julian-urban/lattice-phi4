import numpy as np
import copy


class Lattice:
    def __init__(self, N, d, k, l):
        self.N = N
        self.d = d
        self.shape = [N for _ in range(d)]
        self.k = k
        self.l = l
        
        self.phi = np.random.randn(*self.shape)
        self.action = self.get_action()
    
    def get_action(self):
        action = (1 - 2 * self.l) * self.phi**2 + self.l * self.phi**4

        for mu in range(self.d):
            action += -2. * self.k * self.phi * np.roll(self.phi, 1, mu)

        return action.sum()

    def get_local_action(self, xyz):
        action = (1 - 2 * self.l) * self.phi[xyz]**2 + self.l * self.phi[xyz]**4

        for mu in range(self.d):
            hop = np.zeros((self.d, 1), dtype=int)
            hop[mu,0] = 1
            xyz_plus = tuple(map(tuple, ((np.array(xyz) + hop) % self.N)))
            xyz_minus = tuple(map(tuple, ((np.array(xyz) - hop) % self.N)))
            action += -2. * self.k * self.phi[xyz] * (self.phi[xyz_plus] + self.phi[xyz_minus])

        return action
    
    def get_drift(self):
        drift = 2 * self.phi * (2 * self.l * (1 - self.phi**2) - 1)

        for mu in range(self.d):
            drift += 2. * self.k * (np.roll(self.phi, 1, mu) + np.roll(self.phi, -1, mu))

        return drift
    
    def get_hamiltonian(self, chi, action):
        return 0.5 * np.sum(chi**2) + action

    def metropolis(self, sigma=1.):
        xyz = tuple(map(tuple, np.random.randint(0, self.N, (self.d,1))))
        phi_0 = self.phi[xyz]
        S_0 = self.get_local_action(xyz)
        
        self.phi[xyz] += sigma * np.random.randn()

        dS = self.get_local_action(xyz) - S_0

        if dS > 0:
            if np.random.rand() >= np.exp(-dS):
                self.phi[xyz] = phi_0

                return False
        return True
            
    def langevin(self, dt=0.01):
        chi = np.random.randn(*self.shape)

        self.phi += (dt * self.get_drift() +
                     np.sqrt(dt) * chi)

        return True

    def hmc(self, n_steps=100):
        dt = 1 / n_steps
        phi_0 = copy.deepcopy(self.phi)
        chi = np.random.randn(*self.shape)

        S_0 = self.action
        H_0 = self.get_hamiltonian(chi, S_0)

        chi += 0.5 * dt * self.get_drift()

        for i in range(n_steps):
            self.phi += dt * chi

            if i == n_steps-1:
                chi += 0.5 * dt * self.get_drift()
            else:
                chi += dt * self.get_drift()

        self.action = self.get_action()
        dH = self.get_hamiltonian(chi, self.action) - H_0

        if dH > 0:
            if np.random.rand() >= np.exp(-dH):
                self.phi = phi_0
                self.action = S_0

                return False
        return True
