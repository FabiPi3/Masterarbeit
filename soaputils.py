from __future__ import print_function, division

from dscribe.descriptors import SOAP
from sklearn.cluster import KMeans
import struct_statistics as stst
import random
import os
import gc
import time

import matplotlib as matplotlib
import matplotlib.pyplot as p
import numpy as np
from numpy.linalg import norm, svd
from math import pi

import ase
from ase import units, Atom, Atoms
from ase.build import bulk
from ase.visualize import view
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin

from asap3 import Atoms, EMT, units
from asap3.analysis.rdf import RadialDistributionFunction
from pyqstem import PyQSTEM

from skimage import data
from skimage.transform import rotate
import scipy.spatial.distance as sd
import scipy.interpolate as sci
import scipy.ndimage as scnd
import scipy.ndimage.filters as filters


def rand_pos(atoms_in, seed=None): # function to randomize atom positions. New positions will be inside Unit cell
    atoms = atoms_in.copy()
    pos = atoms.get_positions()
    shape = pos.shape
    np.random.seed(seed)
    ran_pos = np.random.random_sample(shape)
    cell = atoms.get_cell()
    for i in np.arange(shape[0]):
        pos[i] = np.matmul(cell,ran_pos[i])
    atoms.set_positions(pos)
    return atoms

def gen_struct(Natoms, elements, ratios=None, dens=0.06, length=None, seed=None, pbc=True): #function to generate a random atom structure
    '''
    input arguments: Natoms: Number of Atoms
    elements: array of element symbols or numbers that should be in the structure
    ratios: array which specifies the ratios of the elements, can change the total number of Atoms because of round effects, default: every element same ratio
    dens: default 0.06, spicifies the density in the unit cell, only used if length is None
    length: specifies the length of the cubic unit cell
    seed: gives the seed for the randomness of the atom positions
    pbc: periodic boundary conditions True or False
    '''
    assert Natoms >= 9, "Number of Atoms to low. Natoms={}".format(Natoms)
    Ntypes = len(elements) # Number of element types
    if np.any(ratios == None): #default ratios
        ratios = np.ones(Ntypes)
    assert np.shape(elements)==np.shape(ratios), "wrong shape of ratios, must have same size as elements."
    ratios = Natoms*np.array(ratios)/np.sum(np.array(ratios))
    ratios = ratios.astype(int)
    struct = ase.Atoms(elements[0])
    del struct[0]
    if length == None: #default length
        length = np.power(Natoms/dens,1/3)
    for j in np.arange(Ntypes): #fill the structure with atoms
        for i in range(ratios[j]):
            struct.append(ase.atom.Atom(elements[j],(0,0,0)))
    struct.set_cell(np.diag((length,length,length)))
    struct = rand_pos(struct,seed=seed) #randomize positions
    if pbc:
        struct.set_pbc(True)
    return struct
    
def showPlot(x_data, y_data=None, hist=False, bins=25, title=None, xlabel=None, ylabel=None, save=None, fontSize=33, angles=1, leg=None):
    p.rcParams['axes.linewidth'] = 3
    p.ticklabel_format(axis='y',style='scientific', scilimits=(-3,3))
    p.rcParams.update({'font.size': fontSize})
    fig, ax = p.subplots(figsize=(30, 15))
    ax.tick_params(which='both', length=7, width=3)
    if hist:
        p.hist(x_data, bins=bins)
        #p.hist(x_data, bins=bins,range=(0,182), align='left')
        if angles==1:
            p.xticks(ticks=np.arange(0, 190, 10), labels=(0,'','',30,'','',60,'','',90,'','',120,'','',150,'','',180))
    else:
        p.plot(x_data, y_data, linewidth=3)
    if title is not None:
        p.title(title)
    if xlabel is not None:
        p.xlabel(xlabel)
    if ylabel is not None:
        p.ylabel(ylabel)
    if leg is not None:
        p.legend(leg, fontsize=30, loc=2)
    if save is not None:
        name = save + '.pdf'
        p.savefig(name)
        p.close()
    else:
        p.show()

def angledistr(atoms, N=5, alle=True, lab=None, clus=None):
    '''
    function to calculate the angle distribution of given structure atoms up to N nearest neighbours;
    if all is True, uses all atoms, if not, uses only atoms from one cluster: given by cluster; labels given by lab; lab must have same length as number of atoms in atoms
    '''
    Natoms = len(atoms)
    if not alle:
        assert np.shape(lab)[0]==Natoms, "Labels and atom numbers are not fitting."
    else:
        lab=np.zeros(Natoms)
        clus=0
    #N=5 #number of neighbours used
    anz = int(N*(N-1)/2) #number of angles that have to be calculated
    a = atoms.get_all_distances(mic=True) #matrix with all distances
    posis = atoms.get_positions()
    angles = np.zeros((Natoms*anz))
    count = 0 #just an index
    for m in range(int(Natoms)): #for every atom
        if lab[m]==clus:
            temp = a[m,:] #m'th line
            pos1 = posis[m, :]
            ind = np.argsort(temp)[1:N+1]
            for i in range(N): #for every neighbour
                for j in range(4-i): # with the other neighbours
                    #calculate bonding angle
                    k=i+j+1
                    pos2 = posis[ind[i], :]
                    pos3 = posis[ind[k], :]
                    difpos1 = pos2 - pos1
                    difpos2 = pos3 - pos1
                    norm = np.linalg.norm(difpos1)*np.linalg.norm(difpos2)
                    keks = np.dot(difpos1, difpos2)/norm
                    angle = np.arccos(np.around(keks, decimals=5))*180/np.pi
                    angles[count]=angle #save the angle
                    count += 1
    anglesshort=angles[0:count] # to skip zeros if not used all atoms
    return anglesshort.flatten()
    
def electrondiffraction(atoms, angle1=0, angle2=0, slices=10, sample=400, alpha=1.5, reso=None, dis=False):
    '''
    This function calculates for a given atom structure the diffraction pattern in CBED mode. The structure can be rotated before with angle1 around the x axis and angle2 around the y axis. For siumlation PyQSTEM is used with the multislice algorithm. As further input parameters the number of slices, the number of pixels, the resolution in Angstrom and the convergence angle alpha in mrad can be specified. The waves can be displayed.
    '''
    length = atoms.get_cell()
    a = length.lengths()[0] #lenght of super cell
    #Rotation of atom structure
    if angle1 != 0:
        atoms.rotate(angle1, 'x', center=(a/2, a/2, a/2))
    if angle2 != 0:
        atoms.rotate(angle2, 'y', center=(a/2, a/2, a/2))
    atoms.wrap()
    
    resolution = (reso, reso) #resolution in Angstrom
    samples = (sample,sample) # samples in x and y-direction 
    v0 = 200 # acceleration voltage [kV]
    
    qstem = PyQSTEM('CBED')
    qstem.set_atoms(atoms)
    qstem.build_probe(v0,alpha,samples,resolution=resolution,defocus=-6,astig_mag=0,astig_angle=0)
    input_wave=qstem.get_wave()
    
    cell=atoms.get_cell()
    probe_position=(cell[0,0]/2,cell[1,1]/2) #center the electron beam
    
    qstem.build_potential(int(slices),probe_position=probe_position)
    qstem.run()
    exit_wave=qstem.get_wave()
    
    if dis:
        p.rcParams.update({'font.size': 35})
        fig,(ax1)=p.subplots(1,1, figsize=(21,14))
        input_wave.view(ax=ax1,method='intensity',cmap='bwr',vmin=-3,vmax=20,title='')
        p.tight_layout()
        p.savefig('Input_wave_intensity.pdf')
        p.close()
        fig,(ax1)=p.subplots(1,1, figsize=(21,14))
        input_wave.view(ax=ax1,method='real',cmap='bwr',vmin=-3,vmax=5,title='')
        p.tight_layout()
        p.savefig('Input_wave_real.pdf')
        p.close()
        fig,(ax1)=p.subplots(1,1, figsize=(21,14))
        input_wave.view(ax=ax1,method='phase',cmap='bwr',vmin=-5,vmax=5,title='')
        p.tight_layout()
        p.savefig('Input_wave_phase.pdf')
        p.close()
        fig,(ax1)=p.subplots(1,1, figsize=(21,14))
        exit_wave.view(ax=ax1,method='intensity',cmap='bwr',vmin=-3,vmax=20,title='')
        p.tight_layout()
        p.savefig('Exit_wave_intensity.pdf')
        p.close()
        fig,(ax1)=p.subplots(1,1, figsize=(21,14))
        exit_wave.view(ax=ax1,method='real',cmap='bwr',vmin=-3,vmax=5,title='')
        p.tight_layout()
        p.savefig('Exit_wave_real.pdf')
        p.close()
        fig,(ax1)=p.subplots(1,1, figsize=(21,14))
        exit_wave.view(ax=ax1,method='phase',cmap='bwr',vmin=-5,vmax=5,title='')
        p.tight_layout()
        p.savefig('Exit_wave_phase.pdf')
        p.close()
    
    wave_array = np.fft.fftshift(np.fft.fft2(exit_wave.array)) # Transfer exit wave to diffraction plane by fourier transform
    wave_array = np.abs(wave_array)**2 # Calculate intensity
    return wave_array

def correlo(atoms, dis=False, it=3, rCut=6, alpha=1.5, slices=10, winkel1=45, winkel2=0, xy=1, z=2, out=False, reso=0.15):
    '''
    This function computes for a given atom structure the correlograph. Input parameters:
    dis: wheter to display the first diffraction pattern or not, default not
    it: number of diffraction pattern (dp) to be calculated
    rCut: maximal distance used for SOAP calculations
    alpha: convergence angle in mrad
    slices: number of slices for diffraction simulation
    winkel1 and winkel2: angles to rotate the structure between the dps
    xy and z: duplications of the structure in xy plane and z direction
    out: wheter the output is only the correlograph or also the bounds of the k axis
    reso: resolution in Angstrom
    '''
    xy=2*xy+1 #for an odd number of duplications, so that the electron beam hits the center of a unit cell
    atoms = atoms*(xy,xy,z)
    a = atoms.get_cell()[0,0] #unit cell length
    sample = int(a/reso) #number of pixels, given through the size of the supercell and the resolution
    dp = np.zeros((it, sample, sample)) #diffraction pattern
    for j in np.arange(it):
        i = int(j)
        dp[j, :, :] = electrondiffraction(atoms, angle1=winkel1, angle2=winkel2, sample=sample, alpha=alpha, slices=slices, reso=reso)
    if dis:
        fig = p.figure(figsize=(64, 48))
        p.imshow((np.log10(1+dp[0,int(sample/3):int(2*sample/3),int(sample/3):int(2*sample/3)])),cmap='gray',interpolation='bilinear',vmin = 0)
        p.colorbar() #Keks
        p.show()

    #corellographs calculation
    NpixX = sample # number of pixels along x 1200
    NpixY = sample # number of pixels along y 1200
    Nk    = int(sample*3/4) # number of k-points in interpolated array 900
    Nphi  = int(Nk/2)  # number of phi-points in interpolated array 450
    Npattern = it # number of diffraction patterns to be simulated 1
    dx = reso #1  # Pixel size along x 1
    dy = reso #1  # pixel size along y 1
    Nsignal = int(round(0.1*NpixX*NpixY))
    dpPol  = np.zeros((Npattern,Nphi,Nk)) # interpolated diffraction intensities in polar coordinates
    
    # generate a reciprocal space grid:
    kx1 = np.fft.fftshift(np.fft.fftfreq(NpixX,dx)) # 1D reciprocal space coordinate along x-direction 
    ky1 = np.fft.fftshift(np.fft.fftfreq(NpixY,dy))
    kX = np.tile(kx1.reshape(1,NpixX),(NpixY,1))
    kY = np.tile(ky1.reshape(NpixY,1),(1,NpixX))

    # generate kp and phi arrays
    kmax= np.max(np.sqrt(kX**2+kY**2))/(1.01*np.sqrt(2))
    phi = np.arange(-np.pi,np.pi,2*np.pi/Nphi)  # does not include +pi
    k   = np.linspace(0.0,kmax,Nk)              # includes kmax

    # generate the grid on which we compute the interpolated data
    k2,phi2 = np.meshgrid(k,phi)
    kXi = k2*np.cos(phi2)
    kYi = k2*np.sin(phi2)
    for ip in np.arange(Npattern):
        interpSpline = sci.RectBivariateSpline(ky1, kx1, dp[ip]) 
        dpPol[ip] = interpSpline(kYi,kXi,grid=False)
    
    correlograph=np.zeros((Npattern,Nphi,Nk))
    for ip in np.arange(Npattern):
        correlograph[ip] = np.real(np.fft.ifft(np.abs(np.fft.fft(dpPol[ip],axis=0))**2,axis=0))
        # Now we need to normalize this correlograph:
        correlograph[ip] /= np.tile(np.mean(correlograph[ip],axis=0), (Nphi, 1))
        #correlograph[ip] /= np.mean(correlograph[ip])
    correlograph -= 1
    
    # cut some of the k axis, for small k because of rCut and for large k because of the polar coordinates
    cutof_k = (0.5/rCut)
    cutof = int(Nk*cutof_k/kmax)
    cutmax = 2/3
    cutmax_pix = int(Nk*cutmax)
    cutmax_k = kmax*cutmax
    correlograph = correlograph[:,:,cutof:cutmax_pix]

    # output and for more than 1 dp calculate average and variance
    if it==1:
        return correlograph, cutof_k, cutmax_k
    else:
        correlographAvg=np.mean(correlograph,axis=0)

        correlographVar=np.zeros(np.shape(correlographAvg))
        for ip in np.arange(np.shape(correlograph)[0]):
            correlographVar += (correlograph[ip]-correlographAvg)**2
        correlographVar /= correlographAvg**2
        if not out:
            return correlographAvg, correlographVar
        else:
            return correlographAvg, correlographVar, cutof_k, cutmax_k

def svd_norm2(pos, s0=0, s1=1, dis=False, full_out=False):
    '''
    This function calculates the error function for the optimization. You can specifie the scale factors and whether the result should be displayed or not.
    '''
    species = ['Cu']
    Natom = 150
    #make from the pos vector an atom structure
    atoms_obj = gen_struct(Natom, elements=species, length=10)
    pos_ini = atoms_obj.get_positions()
    pos = np.reshape(pos, pos_ini.shape)
    atoms_obj.set_positions(pos)
    
    rCut=6.0
    
    #electron diffraction part
    e1 = 0
    e2 = 0
    if s1 != 0:
        correlographAvg, correlographVar = correlo(atoms_obj, rCut=rCut)

        corVarref = np.load('corVar3.npy')
        corAvgref = np.load('corAvg3.npy')

        e1 = np.sum(np.absolute(correlographAvg - corAvgref)**2)/np.sum(np.absolute(corAvgref)**2)
        e2 = np.sum(np.absolute(correlographVar - corVarref)**2)/np.sum(np.absolute(corVarref)**2)

    #SOAP part
    NradBas=5
    Lmax=5

    soap = SOAP(species=species, periodic=True, rcut=rCut, nmax=NradBas, lmax=Lmax)
    soap = soap.create(atoms_obj)
    s = svd(soap, full_matrices=False, compute_uv=False)
    l1 = norm(s,ord=1)/norm(s,ord=2)

    #sum for the result
    p2 = s0*e2
    p1 = s1*l1
    res = e1 + p2 + p1

    if dis:
        print('corAvgdif:', e1)
        print('corVardif:', e2)
        print('L1-Norm:', l1)
        print('Das Gesamtsummenergebnis lautet wie folgt:', res)

    if full_out and s0==0:
        return e1, l1
    elif full_out:
        return e1, e2, l1
    return res
    
def soapkmeans(pos, use_cc=False, n_clusters=4, dis=False):
    '''
    This function calculates the error function for the optimization, but only based on the SOAP output including a KMeans analysis. You can specify whether the result should be displayed or not.
    '''
    species = ['Cu']
    Natom = 150
    #make from the pos vector an atom structure
    atoms_obj = gen_struct(Natom, elements=species, length=10)
    pos_ini = atoms_obj.get_positions()
    pos = np.reshape(pos, pos_ini.shape)
    atoms_obj.set_positions(pos)
    
    #SOAP part
    rCut=6.0
    NradBas=5
    Lmax=5

    soap = SOAP(species=species, periodic=True, rcut=rCut, nmax=NradBas, lmax=Lmax)
    soap = soap.create(atoms_obj)
    #s = svd(soap, full_matrices=False, compute_uv=False)
    #l1 = norm(s,ord=1)/norm(s,ord=2)
    if use_cc:
        cc = np.load('cc.npy')
        soapkmeans = KMeans(n_clusters=n_clusters,init=cc, n_init=1).fit(soap)
        cc = soapkmeans.cluster_centers_
        np.save('cc', cc)
        res = soapkmeans.inertia_
    else:
        soapkmeans = KMeans(n_clusters=n_clusters,init='k-means++').fit(soap)
        cc = soapkmeans.cluster_centers_
        np.save('cc', cc)
        res=0

    if dis:
        print('Das Gesamtsummenergebnis lautet wie folgt:', res)

    return res   
