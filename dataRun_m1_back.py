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
# import solver as solver
import solverExplicitProg as solver
import scipy.stats as sp2
import streamsculptor
from streamsculptor import potential
from gala.units import UnitSystem
from astropy import units as u
from astropy.constants import G
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
usys.G = G.to(u.kpc**3 / (u.Msun * u.Myr**2)).value

### sim config params
simName = "dataRun_m22=1_back_run1"
N = 256
D = 3
data_drops = 100
cf = .1
L = 25/ np.sqrt(3)
dx = L / N
nf = 1
m22 = np.array([1.0])
rhoDM = 1e7
Mtot = rhoDM * L**3
C = au.G*4*np.pi
Tf = 3500.
initial_drop = 0
T_initial = 0

seed_ = 1
sigma_dm = 216. * au.kms2kpcMyr
n_streams = 64
# print(sigma_dm / au.kms2kpcMyr + 5. / 10000.*Tf)
# print(sigma_dm *Tf)



def PlotStuff(rho):
	rho = su.cpuThis(rho)
	fo = pu.FigObj()
	fo.AddPlot(rho)
	fo.show()


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
	s.integrateBackwards = True
	s.shouldStripStars = False

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

	s.r_prog = np.zeros((1,3))
	s.v_prog = np.zeros((1,3))
	s.r_prog[0] = np.load("r_prog1.npy")[-1]
	s.v_prog[0] = np.load("v_prog1.npy")[-1]

	# initialize dynamic variables
	s.set_K()
	Boltzmann_distr(s)

	N_stars = 1
	s.np = N_stars
	s.r = np.zeros( (N_stars, 3) )
	s.v = np.zeros( (N_stars, 3) )
	s.active = np.full(N_stars, True)

	s.AlterAmp()

	# set temperature and orbit info
	s.sigma = sigma_dm 
	s.T_ref = Tf

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	# PlotStuff(np.abs(s.psi[0,N//2])**2)
	s.RunSim()
