from ase.io import read, write
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.visualize import view
from ase import Atoms
import numpy as np
from numpy import *
from numpy.linalg import norm
#from ase.calculators.emt import EMT # trial -->no!
from ase import Atom

from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory

from asap3 import Atoms, EMT, units
from asap3 import *
from asap3.analysis.rdf import RadialDistributionFunction
import pyqstem #?
from pyqstem import PyQSTEM #!!!

from skimage import data
from skimage.transform import rotate
import scipy.interpolate as sci
import scipy.ndimage.filters as filters

#matplotlib inline
import matplotlib as matplotlib
import matplotlib.pyplot as p
#from pyqstem.util import atoms_plot
from ase.build import bulk
import math
from math import pi
import random
import os
import gc

import soaputils as su
import struct_statistics as stst

from dscribe.descriptors import SOAP
from numpy.linalg import norm, svd
import sys

'''
This code pieces are written to show some specific results.
You can give a number as input to choose the part you're interested in.
'''

i = int(sys.argv[1]) # selection parameter

if i==1:
    # here you can enter a structure and the structure will be displayed
    # and its norm, the angle distribution and the RDF
    s1 = sys.argv[2] # scale factor
    name = sys.argv[3]
    
    s1=float(s1)
    path = "res_structs/" + name + ".cfg"
    atoms = ase.io.read(path)
    view(atoms)
    a = atoms.get_cell()[0,0]

    pos = atoms.get_positions()
    pos = pos.flatten()
    res = su.svd_norm2(pos, s1=s1, dis=True)

    angles = su.angledistr(atoms)
    su.showPlot(angles, hist=True, xlabel='Winkel', ylabel='Anzahl')

    binSize=0.12
    numBins=int(a/binSize)
    h,d = stst.computeRDF(atoms,binSize=binSize, numBins=numBins)
    su.showPlot(d,h, xlabel='Abstand in Angstrom', ylabel='Anzahl')

elif i==2:
    # Here you can plot an aquisition plot.
    # Normally you store only the current value, so here the minimum is computed.
    name = sys.argv[2]
    y = np.load(name + '.npy')
    x = range(len(y))
    y2 = y
    for j in x:
        if j != 0:
            if y2[j]>y2[j-1]:
                y2[j]=y2[j-1]

    p.plot(x,y2)
    #p.ylim(1, 10)
    p.title('Acquisition ' + name)
    p.xlabel('Evaluation')
    p.ylabel('best Error value')
    p.show()
    
elif i==3:
    # Here you can plot another aquisition plot.
    # You can plot the results of the fmin calculations
    versions = sys.argv[2] # to compare the same calculations
    versions = int(versions)
    p.rcParams.update({'font.size': 25})
    for j in range(versions):
        s1=sys.argv[3]
        s1=float(s1)
        anz = sys.argv[j+4] # how often it was minimized
        anz=int(anz)
        res=np.zeros(anz)
        name='res_structs/turbofmin2_' + str(j+1) + '_s1_' + str(s1)
        for i in range(anz):
            if i==0 and j==0:
                path=name+'.cfg'
                atoms=read(path)
                pos=atoms.get_positions().flatten()
                res[i]=su.svd_norm2(pos, s1=s1)
            elif i==0:
                path='Suppenkasper'
            else:
                path=name+'_new_'+str(i)+'.cfg'
                atoms=read(path)
                pos=atoms.get_positions().flatten()
                res[i]=su.svd_norm2(pos, s1=s1)
        x=np.arange(anz)
        p.plot(x,res)
    p.xlabel('Codeiteration')
    p.ylabel('Ergebnis')
    p.show()
    
elif i==4:
    # Here you can plot a l curve plot
    # tested scale factors:
    s1=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]#, 5.0, 8.0, 20.0, 15.0, 50.0]
    #s1=np.sort(s1)
    # how often they was minimized:
    new_n=[10,10,11,11,11,11,10,11,11]
    # calculate cor norm and soap norm and plot against each other:
    cordif = np.zeros(np.shape(s1))
    soapres = np.zeros(np.shape(s1))
    for i in np.arange(np.shape(s1)[0]):
        name = 'turbofmin2_1_s1_' + str(s1[i]) + '_new_' + str(new_n[i])
        path = "res_structs/" + name + ".cfg"
        atoms = ase.io.read(path)
        pos = atoms.get_positions()
        pos = pos.flatten()
        e1, l1 = su.svd_norm2(pos, s1=s1[i], full_out=True)
        cordif[i] = e1
        soapres[i] = l1

    p.rcParams.update({'font.size': 25})
    fig = p.figure() 
    ax = fig.add_subplot(111) 
    c=cordif
    s=soapres
    fac=['0.001','0.01','0.1','1','10','100','1000','10^4','10^5']

    p.loglog(c, s)
    p.xlabel('cordif')
    p.ylabel('soap')
    count=0
    p.rcParams.update({'font.size': 20})
    for xy in zip(c, s): # for the labelling
        ax.annotate('%s' % fac[count], xy=xy, textcoords='data')
        count+=1
    p.show()
    
elif i==5:
    # Here you can plot the singular values for a given structure.
    name = sys.argv[2]
    path = "res_structs/" + name + ".cfg"
    atoms = ase.io.read(path)
    view(atoms)

    # SOAP calculation:
    periodic_soap = SOAP(
        species=['Cu'],
        rcut=6,
        nmax=5,
        lmax=5,
        periodic=True
    )

    soap_cu = periodic_soap.create(atoms)
    s=svd(soap_cu,compute_uv=False)
    p.rcParams.update({'font.size': 35})
    p.semilogy(s[0:20])
    p.xlabel('Index')
    p.ylabel('Singular Value')
    p.show()
    
else:
    print('Error')
