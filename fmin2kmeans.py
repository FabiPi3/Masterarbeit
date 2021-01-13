import numpy as np
import matplotlib.pyplot as p
import scipy.optimize as so
import soaputils as su
import ase
from ase.io import read, write
from ase.visualize import view

import sys
import time

'''
This code takes a structure and tries to minimize my defined norm with the Nelder-Mead-Algorithm.
As inputs the name of the starting structure is defined and the scale factor s1.
Here the KMeans Norm is minimized.
'''

# inputs to specify the starting structure
alg = sys.argv[1] # algorithm
n = sys.argv[2] # index
s1 = sys.argv[3] # scale factor
s1 = float(s1)
new_n = sys.argv[4] # new index
new_n=int(new_n)
kme=sys.argv[5]
kme=int(kme)

# load starting structure:
if new_n==0:
    atoms = read('res_structs/' + alg + 'pbc2_' + n + "_s1_" + str(s1) + '.cfg')
else:
    if kme==0:
        atoms = read("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + ".cfg")
    else:
        atoms = read("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + "kmeans.cfg")

pos_ini = atoms.get_positions()
pos_ini = pos_ini.flatten()

su.soapkmeans(pos_ini) # for initial cluster centers

t0 = time.time()

def f(x): # define the function, which calculates the norm
    res = su.soapkmeans(x, use_cc=True)
    return res

mini = so.fmin(f, pos_ini, ftol=0.0001, maxiter=500000)

su.svd_norm2(mini, s1=s1, dis=True)

dt = (time.time()-t0)/3600
print('Gedauerte Zeit:', dt)

# save the resulting structure:
pos = np.reshape(mini, (-1,3))
atoms.set_positions(pos)
new_n+=1
write("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + "kmeans.cfg", atoms)
