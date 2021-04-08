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
to execute: python3 fmin2kmeans.py turbo 1 100.0 48 0 20
'''

alg = sys.argv[1] # algorithm
n = sys.argv[2] # index
s1 = sys.argv[3]
s1 = float(s1)
new_n = sys.argv[4] # new index
new_n=int(new_n)
kme=sys.argv[5]
kme=int(kme)
anz=sys.argv[6]
anz=int(anz)

for i in range(anz):
    if new_n==0:
        atoms = read('res_structs/' + alg + 'pbc2_' + n + "_s1_" + str(s1) + '.cfg')
    else:
        if kme==0:
            atoms = read("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + ".cfg")
        else:
            atoms = read("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + "kmeans.cfg")

    pos_ini = atoms.get_positions()
    pos_ini = pos_ini.flatten()

    su.soapkmeans(pos_ini) #for initial cluster centers

    t0 = time.time()

    def f(x):
        res = su.soapkmeans(x, use_cc=True)
        return res

    mini = so.fmin(f, pos_ini, ftol=0.0001, maxiter=500000)

    su.soapkmeans(mini, dis=True)

    dt = (time.time()-t0)/3600
    print('Gedauerte Zeit:', dt)

    pos = np.reshape(mini, (-1,3))
    atoms.set_positions(pos)
    kme=1
    new_n+=1
    write("res_structs/" + alg + "fmin2_" + n + "_s1_" + str(s1) + "_new_" + str(new_n) + "kmeans.cfg", atoms)
