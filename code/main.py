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
cfgs = []

for i in tqdm(range(1000)):
	# lattice.langevin()
	lattice.hmc()

	cfgs.append(copy.deepcopy(lattice.phi))

cfgs = np.array(cfgs)
axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
np.savetxt("mag.dat", cfgs.mean(axis=axis))


print("\nrecording...\n")
cfgs = []

for i in tqdm(range(10000)):
    # lattice.langevin()
    lattice.hmc()

    if i % 10:
        cfgs.append(copy.deepcopy(lattice.phi))

cfgs = np.array(cfgs)
print("\ndone.")


print("\ncalculating observables...\n")

mag_mean, mag_err = get_mag(cfgs)
print("M =", mag_mean, "+/-", mag_err)

mag_abs_mean, mag_abs_err = get_abs_mag(cfgs)
print("|M| =", mag_abs_mean, "+/-", mag_abs_err)

chi2_mean, chi2_err = get_chi2(cfgs)
print("chi2 =", chi2_mean, "+/-", chi2_err)

corr_func = get_corr_func(cfgs)
np.savetxt("corr_func.dat", corr_func)
