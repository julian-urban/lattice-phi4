import numpy as np
import copy
from tqdm import tqdm

from lattice import Lattice
from functions import *


N = 32
d = 2
k = 0.3
l = 0.02

lattice = Lattice(N, d, k, l)


print("burn in...\n")
mag = []

for i in tqdm(range(1000)):
    # lattice.langevin()
    # lattice.hmc()
    lattice.metropolis()

    mag.append(lattice.phi.mean())

np.savetxt("mag.dat", np.array(mag))


print("\nrecording...\n")
cfgs = []
n_accepted = 0
n_steps = 10000

for i in tqdm(range(n_steps)):
    # n_accepted += lattice.langevin()
    # n_accepted += lattice.hmc()
    n_accepted += lattice.metropolis()

    if i % 10 == 0:
        cfgs.append(copy.deepcopy(lattice.phi))

cfgs = np.array(cfgs)
print("\ndone.")
print("accept rate:", n_accepted / n_steps)

print("\ncalculating observables...\n")

mag_mean, mag_err = get_mag(cfgs)
print("M =", mag_mean, "+/-", mag_err)

mag_abs_mean, mag_abs_err = get_abs_mag(cfgs)
print("|M| =", mag_abs_mean, "+/-", mag_abs_err)

chi2_mean, chi2_err = get_chi2(cfgs)
print("chi2 =", chi2_mean, "+/-", chi2_err)

corr_func = get_corr_func(cfgs)
np.savetxt("corr_func.dat", corr_func)
