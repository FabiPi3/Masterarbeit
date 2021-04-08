import ase
from ase.io import read, write
from ase.visualize import view
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory

from asap3 import Atoms, EMT, units
from asap3.analysis.rdf import RadialDistributionFunction

import matplotlib.pyplot as p
import numpy as np
import soaputils as su
import struct_statistics as stst

a=10
'''
Natoms = 150 # number of Atoms
#start structure
atoms = su.gen_struct(Natoms, seed=50, elements=['Cu'], length=a)
atoms.set_calculator(EMT()) # ASAP calculator, much faster as the calculator from ase

T = 300  # temperature in Kelvin
dyn = Langevin(atoms, 5 * units.fs, T * units.kB, 0.002) # MD

def printenergy(a=atoms):
    epot = a.get_potential_energy() / Natoms
    ekin = a.get_kinetic_energy() / Natoms
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%4.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    
dyn.attach(printenergy, interval=100000) #attach print function

# save the atom positions at some point:
traj = Trajectory('atoms2.traj', 'w', atoms) #ase gui atoms.traj
dyn.attach(traj.write, interval=500)

printenergy()
# run MD with setting velocities to zero every step
for i in range(5000000):
    atoms.set_velocities(np.zeros((Natoms, 3)))
    dyn.run(1)

view(atoms)
write('res_structs/genref3.cfg', atoms)'''
atoms=read('res_structs/genref3.cfg')

a_g=3*a
reso=0.15
sample = int(a_g/reso)
atoms_g = atoms*(3,3,2)
#su.electrondiffraction(atoms_g, sample=sample, alpha=1.5, reso=reso, dis=True)
print('a')

# calculate correlographs from the reference structure
# use high number of iterations for a good variance
p.rcParams.update({'font.size': 33})
correlographAvg, correlographVar, kmin, kmax = su.correlo(atoms, dis=True, it=35, winkel1=10, out=True)

#np.save('corAvg3', correlographAvg)
#np.save('corVar3', correlographVar)
#correlographAvg = np.load('corAvg3.npy')
#correlographVar = np.load('corVar3.npy')

fig = p.figure(figsize=(64, 48))
p.imshow(np.log(1+np.abs(correlographAvg)),extent=[kmin,kmax,2*np.pi,0], vmin=0, vmax=0.5)
p.axis('auto'); p.axis('tight'); p.colorbar()
p.xlabel('k in inversen Angstrom')
p.ylabel('Winkel in Radiant')
p.show()

fig = p.figure(figsize=(64, 48))
p.imshow(np.log(1+np.abs(correlographVar)),extent=[kmin,kmax,2*np.pi,0], vmin=0, vmax=15)
p.axis('auto'); p.axis('tight'); p.colorbar()
p.xlabel('k in inversen Angstrom')
p.ylabel('Winkel in Radiant')
p.show()
'''
angles = su.angledistr(atoms)
su.showPlot(angles, hist=True, xlabel='Winkel', ylabel='Anzahl')

binSize=0.12
numBins=int(a/binSize)
h,d = stst.computeRDF(atoms,binSize=binSize, numBins=numBins)
su.showPlot(d,h, xlabel='Abstand in Angstrom', ylabel='Anzahl')

'''
