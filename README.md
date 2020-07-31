# Lattice Scalar Field Theory Simulation

A minimal python example of a lattice simulation for real, scalar phi-fourth theory on a two-dimensional periodic square lattice in the dimensionless formulation. Details about the theory can be found e.g. in [arxiv:1705.06231](https://arxiv.org/abs/1705.06231). Currently implemented algorithms are the Hybrid/Hamiltonian Monte Carlo as well as a first-order Langevin update.

`main.py` contains a basic burn-in loop for performing 1000 HMC updates with a choice of action parameters in the broken phase, on a 32x32 lattice. The value of the field magnetization at every point in the Markov chain is stored in a text file. Plotting this file, one can observe equilibration to a non-zero expectation value of the condensate:

![alt text](mag.png "mag.png")

Note that equilibration can take longer in some cases and should be carefully monitored.

Following the burn-in phase, the simulation is run for an additional 10000 steps and every 10th configuration is recorded. Some observables are calculated with the resulting dataset, errors are obtained using the statistical jackknife method. The two-point correlation function takes the following shape:

![alt text](corr_func.png "corr_func.png")

The theory belongs to the Ising universality class. As such, in d>1 one observes a phase transition associated with spontaneous breaking of Z2 symmetry. Calculating expectation values of the magnetization for various combinations of couplings, we can visualize the phase boundary:

![alt text](phase_diagram.png "phase_diagram.png")
