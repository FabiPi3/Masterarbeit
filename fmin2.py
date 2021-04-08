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
fmin is without randomness
to execute: python3 fmin2fmin.py turbo 1 0.01 1 
'''

alg = sys.argv[1] # algorithm
n = sys.argv[2] # index
s1 = sys.argv[3]
s1 = float(s1)
new_n = sys.argv[4] # start index
new_n=int(new_n)
anz = sys.argv[5] # how often
anz=int(anz)

def f(x):
    res = su.svd_norm2(x, s1=s1)
    return res

for wdhs in range(anz):
    if new_n==0:
        atoms = read('res_structs/' + alg + 'pbc2_' + n + "_s1_" + str(s1) + '.cfg')
    else:
        atoms = read("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + ".cfg")

    pos_ini = atoms.get_positions()
    pos_ini = pos_ini.flatten()

    t0 = time.time()

    mini = so.fmin(f, pos_ini, ftol=0.0001, maxiter=10000)

    su.svd_norm2(mini, s1=s1, dis=True)

    dt = (time.time()-t0)/3600
    print('Gedauerte Zeit:', dt)

    pos = np.reshape(mini, (-1,3))
    atoms.set_positions(pos)
    new_n+=1
    write("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + ".cfg", atoms)
