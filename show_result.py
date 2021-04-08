from ase.io import read, write
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.visualize import view
from ase import Atoms
from ase.build import bulk
import numpy as np
from numpy import *
from numpy.linalg import norm, svd
#from ase.calculators.emt import EMT # trial -->no!
from ase import Atom
from dscribe.descriptors import SOAP
from sklearn.cluster import KMeans
import time as t

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

i = int(sys.argv[1]) #TODO: try catch
#name = sys.argv[2]

if i==1:
    ## to execute: python3 show_result.py 1 0.01 1 pbc
    s1 = sys.argv[2]
    name = sys.argv[3]
    
    s1=float(s1)
    path = "Structs/" + name + ".cfg"
    atoms = ase.io.read(path)
    #view(atoms)
    a = atoms.get_cell()[0,0]

    pos = atoms.get_positions()
    pos = pos.flatten()
    res = su.svd_norm2(pos, s1=s1, dis=True)
    print(res)

    angles = su.angledistr(atoms)
    #su.showPlot(angles, hist=True, xlabel='Winkel in Grad', ylabel='Anzahl')

    binSize=0.09
    numBins=int(a/binSize/2) #because of pbc reasons
    h,d = stst.computeRDF(atoms,binSize=binSize, numBins=numBins)
    #su.showPlot(d,h, xlabel='Abstand in Angstrom', ylabel='Anzahl', fontSize=31)

elif i==2:
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
    
elif i==3: # aquisition plot python3 show_result.py 3 4 100.0 48 26 30 24
    versions = sys.argv[2]
    versions = int(versions)
    p.rcParams.update({'font.size': 33})
    p.rcParams['axes.linewidth'] = 3
    fig, ax = p.subplots(figsize=(30, 15))
    ax.tick_params(which='both', length=7, width=3)
    for j in range(versions):
        s1=sys.argv[3]
        s1=float(s1)
        anz = sys.argv[j+4]
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
                res[i]=115
            else:
                path=name+'_new_'+str(i)+'.cfg'
                atoms=read(path)
                pos=atoms.get_positions().flatten()
                res[i]=su.svd_norm2(pos, s1=s1)
            #new
            if i!=0 and i!=1 and res[i]>res[i-1]:
                res[i]=res[i-1]
            #new end
        x=np.arange(anz)
        p.plot(x,res, linewidth=3, label='Versuch ' + str(j+1))
        p.scatter(x,res,marker="x", s=200, linewidth=3)
    p.xlabel('Codeiteration')
    p.ylabel('N_{ges}')
    p.ylim(101,106)
    p.legend()
    p.show()
    
elif i==4: #plot l curve
    s1=[0.1, 1.0, 10.0, 100.0, 1000.0]#, 5.0, 8.0, 20.0, 15.0, 50.0]
    #s1=[2.5, 5.0, 8.0, 10.0, 100.0, 1000.0, 10000.0]#, 5.0, 8.0, 20.0, 15.0, 50.0]
    #s1=np.sort(s1)
    new_n=[11,11,11,48,10]
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

    p.rcParams.update({'font.size': 31})
    p.rcParams['axes.linewidth'] = 3
    fig, ax = p.subplots(figsize=(30, 15))
    ax.tick_params(which='both', length=7, width=3)
    c=cordif
    s=soapres
    #fac=['0.1','1','10','100','10³']

    p.loglog(c, s, linewidth=3)
    p.scatter(c,s,marker="x", s=200, linewidth=3)
    p.yticks(ticks=(1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16),labels=(1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16))
    p.xlabel('N_{KOR}')
    p.ylabel('N_{SOAP}')
    #count=0
    #p.rcParams.update({'font.size': 20})
    #for xy in zip(c, s):
    #    ax.annotate('%s' % fac[count], xy=xy, textcoords='data', size=25)
    #    count+=1
    p.show()
    
elif i==5: #plot singular values
    #s1 = sys.argv[2]
    #name = sys.argv[2]
    #s1=float(s1)
    path = "Structs/Turboopt.cfg"
    atoms = ase.io.read(path)
    path = "Structs/TurboKmeansopt.cfg"
    atoms2 = ase.io.read(path)
    #view(atoms)
    
    #nacl = bulk('NaCl', 'zincblende', a=3.6, cubic=True)
    #print(copper.get_pbc())
    periodic_soap = SOAP(
        species=['Cu'],
        rcut=6,
        nmax=5,
        lmax=5,
        periodic=True
    )

    soap_cu1 = periodic_soap.create(atoms)
    s1=svd(soap_cu1,compute_uv=False)
    soap_cu2 = periodic_soap.create(atoms2)
    s2=svd(soap_cu2,compute_uv=False)
    p.rcParams.update({'font.size': 33})
    p.rcParams['axes.linewidth'] = 3
    fig, ax = p.subplots(figsize=(30, 15))
    ax.tick_params(which='both', length=7, width=3)
    x=np.arange(0,20)
    p.semilogy(s1[0:20], linewidth=3, label="P_{TuRBO}")
    p.semilogy(s2[0:20], linewidth=3, label="P_{KMeans}")
    p.scatter(x,s1[0:20],marker="x", s=200, linewidth=3)
    p.scatter(x,s2[0:20],marker="x", s=200, linewidth=3)
    p.xticks(np.arange(0,22,2))
    p.xlabel('Index des Wertes')
    p.ylabel('Singulärwert')
    p.legend()
    p.show()
    
elif i==6: # aquisition plot kmeans python3 show_result.py 3 4 100.0 41 16 18 16
    p.rcParams.update({'font.size': 25})
    s1=sys.argv[2]
    s1=float(s1)
    start = sys.argv[3]
    start=int(start)
    anz = sys.argv[4]
    anz=int(anz)
    res=np.zeros(anz)
    name='res_structs/turbofmin2_1_s1_' + str(s1) + '_new_'
    for i in range(anz):
        path=name+str(start+i)+'kmeans.cfg'
        atoms=read(path)
        pos=atoms.get_positions().flatten()
        #res[i]=su.svd_norm2(pos) #TODO
    x=np.arange(anz)
    p.plot(x,res)
    p.xlabel('Codeiteration')
    p.ylabel('Ergebnis')
    p.show()
    
elif i==7: #einzelverteilung der rdf und winkel für cluster
    #iteration over clusters:
    atoms = read('res_structs/turbofmin2_1_s1_100.0_new_68kmeans.cfg')
    lab=np.load('lab.npy')
    for i in range(4):
        #print('Cluster:',i)
        angles = su.angledistr(atoms, alle=False, lab=lab, clus=i)
        #su.showPlot(angles, hist=True, xlabel='Winkel in Grad', ylabel='Häufigkeit',leg=['Cluster ' + str(i+1)])
    a=atoms.get_cell()[0,0]
    #prepare structure:
    numb = atoms.get_atomic_numbers()
    numb+=lab
    atoms.set_atomic_numbers(numb)
    #view(atoms)
    
    binSize=0.09
    numBins=int(a/binSize/2) #because of pbc reasons
    h,d = stst.computeRDF(atoms,binSize=binSize, numBins=numBins)
    
    unique, counts = np.unique(lab, return_counts=True)
    ind=0
    for i in range(4):
        for j in np.arange(i,4):
            scafac=counts[i]*counts[j]/1000
            h[:,ind]=h[:,ind]/scafac
            ind+=1
    
    #su.showPlot(d,h, xlabel='Abstand in Angstrom', ylabel='Anzahl', leg=['0-0','0-1','0-2','0-3','1-1','1-2','1-3','2-2','2-3','3-3'])
    su.showPlot(d,np.array((h[:,0],h[:,1],h[:,2],h[:,3])).transpose(), xlabel='Abstand in Angström', ylabel='Häufigkeit', leg=['1-1','1-2','1-3','1-4'], fontSize=31)
    su.showPlot(d,np.array((h[:,1],h[:,4],h[:,5],h[:,6])).transpose(), xlabel='Abstand in Angström', ylabel='Häufigkeit', leg=['2-1','2-2','2-3','2-4'], fontSize=31)
    su.showPlot(d,np.array((h[:,2],h[:,5],h[:,7],h[:,8])).transpose(), xlabel='Abstand in Angström', ylabel='Häufigkeit', leg=['3-1','3-2','3-3','3-4'], fontSize=31)
    su.showPlot(d,np.array((h[:,3],h[:,6],h[:,8],h[:,9])).transpose(), xlabel='Abstand in Angström', ylabel='Häufigkeit', leg=['4-1','4-2','4-3','4-4'], fontSize=31)

elif i==8:
    name1=sys.argv[2]
    name2=sys.argv[3]
    name3=name2+'out'
    x=np.load(name1+'.npy')
    y=np.load(name2+'.npy')
    y2=np.load(name3+'.npy')
    p.rcParams['axes.linewidth'] = 3
    p.rcParams.update({'font.size': 31})
    fig, ax = p.subplots(figsize=(30, 15))
    ax.tick_params(which='both', length=7, width=3)
    p.ticklabel_format(axis='y',style='scientific', scilimits=(-2,2))
    p.plot(x, y, linewidth=3, label='mit')
    p.plot(x, y2, linewidth=3, label='ohne')
    p.scatter(x,y,marker="x", s=200, linewidth=3)
    p.scatter(x,y2,marker="x", s=200, linewidth=3)
    p.xticks(ticks=np.arange(0.0,0.5,0.025),labels=(0.0,'','','',0.1,'','','',0.2,'','','',0.3,'','','',0.4,'','',''))
    p.xlabel('Maximale Verschiebung in Angström')
    p.ylabel('Laufzeit in Sekunden')
    p.legend()
    p.show()

elif i==9:
    name=sys.argv[2]
    x=np.arange(0.4,0.95,0.025)
    y=np.load(name+'.npy')
    p.rcParams['axes.linewidth'] = 3
    p.rcParams.update({'font.size': 31})
    fig, ax = p.subplots(figsize=(30, 15))
    ax.tick_params(which='both', length=7, width=3)
    p.ticklabel_format(axis='y',style='scientific', scilimits=(-2,2))
    p.plot(x, y, linewidth=3)
    p.scatter(x,y,marker="x", s=200, linewidth=3)
    p.xticks(ticks=x,labels=(0.4,'','','',0.5,'','','',0.6,'','','',0.7,'','','',0.8,'','','',0.9,''))
    p.xlabel('Maximale Verschiebung in Angström')
    p.ylabel('relative Fehlerrate')
    p.show()

elif i==10: #plot singular values
    a=10 # atom spacing
    # Salt
    atoms=bulk('Na', 'sc', a=a)
    atoms.append(ase.atom.Atom('Cl',(a/2,0,0)))
    atoms.append(ase.atom.Atom('Na',(a/2,a/2,0)))
    atoms.append(ase.atom.Atom('Cl',(a/2,a/2,a/2)))
    atoms.append(ase.atom.Atom('Na',(a/2,0,a/2)))
    atoms.append(ase.atom.Atom('Cl',(0,a/2,0)))
    atoms.append(ase.atom.Atom('Na',(0,a/2,a/2)))
    atoms.append(ase.atom.Atom('Cl',(0,0,a/2)))
    NaCl = atoms*(4,4,4) # replicate unit cell
    
    periodic_soap = SOAP(
        species=['Na', 'Cl'],
        rcut=6,
        nmax=5,
        lmax=5,
        periodic=True
    )

    soap = periodic_soap.create(NaCl)
    s1=svd(soap,compute_uv=False)
    p.rcParams.update({'font.size': 33})
    p.rcParams['axes.linewidth'] = 3
    fig, ax = p.subplots(figsize=(30, 15))
    ax.tick_params(which='both', length=7, width=3)
    x=np.arange(0,20)
    p.semilogy(s1[0:20], linewidth=3, label='NaCl')
    p.scatter(x,s1[0:20],marker="x", s=200, linewidth=3)
    p.xticks(np.arange(0,22,2))
    p.xlabel('Index des Wertes')
    p.ylabel('Singulärwert')
    p.legend()
    p.show()

else:
    print('Error')
