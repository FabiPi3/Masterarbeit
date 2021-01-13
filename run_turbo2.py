import numpy as np
import ase
from ase.io import read, write
import matplotlib
import matplotlib.pyplot as p
import soaputils as su
import time

from turbo import Turbo1
import math
import torch
import sys

import scipy.optimize as so

'''
This code computes the Bayesian Optimization for given boundary conditions of a structure for a given error function using the TuRBO implementation.
As inputs the scale factor has to be specified, and a factor to specify how much iterations should be performed.
'''

# input stuff:
n = sys.argv[1] # just an index
s1 = sys.argv[2] # scalefactor
s1=float(s1)
print(s1)
it_fac = sys.argv[3] # factor of iterations
it_fac = int(it_fac)

# specify the boundary conditions:
Natoms = 150
atoms = su.gen_struct(Natoms, seed=50, elements=['Cu'], length=10)
cell_len = atoms.get_cell()[0,0]

# define the error function to minimize:
class svdnorm:
    def __init__(self, dim=3*Natoms): # every atom has 3 dimensions to move
        self.dim = dim
        # boundary conditions part 2: the atoms shall stay inside the unit cell:
        self.lb = np.zeros(dim)
        self.ub = cell_len * np.ones(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        res = su.svd_norm2(x, s1=s1)
        return res

f = svdnorm()

t0 = time.time() # for time measurement

# specify the optimization parameters:
turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20*it_fac,  # Number of initial bounds from an Latin hypercube design 20
    max_evals = 100*it_fac,  # Maximum number of evaluations 100
    batch_size=10*it_fac,  # How large batch size TuRBO uses 10
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo1.optimize() # performe the optimization

X = turbo1.X  # Evaluated points
fX = turbo1.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best]

print("Best value found:")
print(f_best)

fig = p.figure(figsize=(7, 5))
matplotlib.rcParams.update({'font.size': 16})
p.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
p.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
p.xlim([0, len(fX)])
p.tight_layout()
p.savefig('turboacq2_' + n + "_s1_" + str(s1) + '.png')

atoms_res = atoms.copy()
atoms_res.set_positions(np.reshape(x_best,(-1,3)))

# the resulting struct is saved in a folder
filename = "turbopbc2_" + n + "_s1_" + str(s1) + ".cfg"
ase.io.write("res_structs/" + filename, atoms_res)

t1 = time.time()
dt = (t1 - t0)/3600
print(dt) # print elapsed time
