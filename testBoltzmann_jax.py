# pylint: disable=C,W
# import numpy as np
# import numpy.linalg as npl
gpu = True
# if gpu:
# 	try:
# 		import cupy as np
# 		import cupy.linalg as npl
# 		import cupy as np
# 	except:
# 		pass
# import cupy as np
# import cupy.linalg as npl
import jax.numpy as np 
import jax as jp
import jax.numpy.linalg as npl 
import astroUtils as au
import gridUtils as gu
import plotUtils as pu
import time 
import sysUtils as su
import sys
sys.path.insert(1, 'Solvers')
import solverJax as solver
import scipy.stats as sp2
import streamsculptor
from streamsculptor import potential
from gala.units import UnitSystem
from astropy import units as u
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)

### sim config params
simName = "testRun2"
N = 128
D = 3
data_drops = 10
cf = .1
L = 20*2 / np.sqrt(3)
dx = L / N
nf = 1
m22 = np.array([5.1])
Mtot = 2e8
C = au.G*4*np.pi
Tf = 1000.
initial_drop = 0
T_initial = 0

seed_ = 0
key = jp.random.PRNGKey(0)

sigma_dm = 2. * au.kms2kpcMyr
n_streams = 64
print(sigma_dm / au.kms2kpcMyr + 5. / 10000.*Tf)
print(sigma_dm *Tf)

# returns a random variable in a ball
def randomInBall(Npoints):  
	"""
	returns random variables in a 3 ball

	:Npoints: int, number of points to return
	"""  
	x = jp.random.normal(key,Npoints)
	y = jp.random.normal(key,Npoints)
	z = jp.random.normal(key,Npoints)

	points = np.zeros((Npoints, 3))
	points[:,0] = x
	points[:,1] = y    
	points[:,2] = z

	norm  = 1./npl.norm(points, axis = 1)
	points = np.einsum("ij,i->ij",points, norm )
	#mag = np.random.exponential(size = Npoints)
	mag = sp2.maxwell.rvs(size = Npoints )
	points = np.einsum("ij,i->ij",points, mag )

	if gpu:
		points = su.gpuThis(points)

	return points


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
	psi_k *= np.exp(-1j*jp.random.uniform(key,minval = 0,
		maxval = 2*np.pi,shape = psi_k.shape))
	# s.psi[0] = s.GetFFt(psi_k, Forward=False, NoFieldDimension = True)
	# s.psi[0,:,:,:] /= np.sqrt(np.sum(np.abs(s.psi[0,:,:,:])**2)*dx**3)
	# s.psi[0,:,:,:] *= np.sqrt(Mtot)
	s.psi = s.psi.at[0].set(s.GetFFt(psi_k, Forward=False, NoFieldDimension = True))
	# s.psi.at[0].set(s.psi.at[0] / np.sqrt(np.sum(np.abs(s.psi)**2)*dx**3))
	s.psi /= np.sqrt(np.sum(np.abs(s.psi)**2)*dx**3)
	s.psi *= np.sqrt(Mtot)
	print(s.psi.device)

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

	# initialize dynamic variables
	s.set_K()
	Boltzmann_distr(s)
	s.r_prog = np.load("r_prog1.npy")
	s.v_prog = np.load("v_prog1.npy")
	s.t_prog = np.load("t_prog1.npy")

	s.r_stars = np.load("r_stars1.npy") # first half of this array is leading arm, second is trailing arm
	s.v_stars = np.load("v_stars1.npy")
	s.t_stars = np.load("t_strip1.npy")
	N_stars = len(s.r_stars)
	s.np = N_stars
	s.r = np.zeros( (N_stars, 3) )
	s.v = np.zeros( (N_stars, 3) )
	# s.active = np.empty(N_stars, dtype = np.bool_)
	# s.active.fill(False)
	s.active = np.full(N_stars, False)

	# set temperature and orbit info
	s.sigma = sigma_dm 
	# s.v_bulk = np.zeros(3)
	# s.v_bulk[2] = sigma_dm 
	# s.v_bulk[1] = 5*L / Tf

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	# PlotStuff(np.abs(s.psi[0,N//2])**2)
	s.RunSim()
