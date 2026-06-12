# pylint: disable=C,W
# import numpy as np
# import numpy.linalg as npl
gpu = True
if gpu:
	try:
		import cupy as np
		import cupy.linalg as npl
		import cupy as np
	except:
		pass
# import jax.numpy as np 
# import jax as jp
# import jax.numpy.linalg as npl 
import astroUtils as au
import gridUtils as gu
import plotUtils as pu
import time 
import sysUtils as su
import sys
sys.path.insert(1, 'Solvers')
import solverForwardTrack as solver
import scipy.stats as sp2
import streamsculptor
from streamsculptor import potential
from gala.units import UnitSystem
from astropy import units as u
from astropy.constants import G
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
usys.G = G.to(u.kpc**3 / (u.Msun * u.Myr**2)).value

### sim config params
simName = "testRun_forwards_m22=1"
refSim = "testRun_backwards_m22=1"
N = 256
D = 3
data_drops = 20
cf = .1
L = 25/ np.sqrt(3)
dx = L / N
nf = 1
m22 = np.array([1])
rhoDM = 1e7
Mtot = rhoDM * L**3
C = au.G*4*np.pi
Tf = 3500.
initial_drop = 0
T_initial = 0

seed_ = 0

sigma_dm = 216. * au.kms2kpcMyr
n_streams = 64



def Boltzmann_distr(s):
	s.psi = np.zeros((nf,N,N,N)) + 0j
	hbar_ = au.h_tilde(m22)[0]
	v2 = s.K*hbar_**2

	# make un-normed version in velocity space
	psi_k = np.exp(-v2 / 2. / sigma_dm**2) + 0j 
	# give random phases
	psi_k *= np.exp(-1j*np.random.uniform(0, 2*np.pi, size = psi_k.shape))
	s.psi[0] = s.GetFFt(psi_k, Forward=False, NoFieldDimension = True)
	s.psi[0,:,:,:] /= np.sqrt(np.sum(np.abs(s.psi[0,:,:,:])**2)*dx**3)
	s.psi[0,:,:,:] *= np.sqrt(Mtot)


def SetICs():
	s = solver.Solver()

	# sim params
	s.simName = simName
	s.N = N 
	s.mp = 0
	s.data_drops = data_drops
	s.initial_drop = initial_drop
	s.T_initial = T_initial
	s.cf = cf
	s.gpu = gpu 
	s.integrateBackwards = False
	s.shouldStripStars = True

	# physics params
	s.L = L
	s.dx = L / N
	s.nf = nf 
	s.D = D 
	s.C = C
	s.Tf = Tf
	s.m22 = m22
	s.hbar_ = au.h_tilde(m22)
	s.pot_MW = potential.GalaMilkyWayPotential(units=usys)

	s.r_prog = np.load("r_prog1.npy")
	s.v_prog = np.load("v_prog1.npy")
	s.t_prog = np.load("t_prog1.npy")

	s.r_stars = np.load("r_stars1.npy") # first half of this array is leading arm, second is trailing arm
	s.v_stars = np.load("v_stars1.npy")
	s.t_stars = np.load("t_strip1.npy")

	# initialize dynamic variables
	s.set_K()

	N_stars = len(s.r_stars)
	s.np = N_stars
	
	s.psi = np.load(f"Data/{refSim}/psi/drop100.npy")
	s.r_perturb = np.load(f"Data/{refSim}/r/drop100.npy")
	s.v_perturb = np.load(f"Data/{refSim}/v/drop100.npy")
	
	s.r = np.zeros( (N_stars, 3) )
	s.v = np.zeros( (N_stars, 3) )
	s.active = np.full(N_stars, False)

	# set temperature and orbit info
	s.sigma = sigma_dm 

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	# PlotStuff(np.abs(s.psi[0,N//2])**2)
	s.RunSim()
