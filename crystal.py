import numpy as np

import ase
from ase import units, Atom, Atoms
from ase.build import bulk
from ase.visualize import view
from ase.io import read, write

import matplotlib as matplotlib
import matplotlib.pyplot as p

import soaputils as su
import struct_statistics as stst

'''
This script is a test for my code. Here the rdf, angle distribution and 
diffraction pattern and correlograph for a simple crystal is computed.
'''

a = 5.0 # cubic unit cell length
atoms = bulk('Cu', 'sc', a=a, cubic=True)
pos=atoms.get_positions()
pos=pos+np.array((2.5,2.5,2.5)) #shift into the middle of the unit cell
atoms.set_positions(pos)
N = 8 # number of atoms in each direction
a1=atoms*(N,N,N)
#view(a1)

#computation of the RDF
h,d = stst.computeRDF(a1, numBins=40)
p.plot(d,h)
p.xlabel('Abstand')
p.ylabel('Anzahl')
p.show()

#computation of the angle distribution
angles = su.angledistr(a1)
p.hist(angles, bins=25)
p.xlabel('Winkel')
p.ylabel('Anzahl')
p.show()

#correlograph calculation
correlograph, kmin, kmax = su.correlo(a1, dis=True, it=1, alpha=0.4, slices=300, winkel1=0, xy=0, z=1, reso=reso[j])

p.imshow(np.log(1+np.abs(correlograph[0,:,:])),extent=[kmin,kmax,2*np.pi,0])#, vmin=-1, vmax=-0.5)
p.axis('auto'); p.axis('tight'); p.colorbar()
p.show()
