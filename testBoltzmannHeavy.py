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
import solver as solver
import scipy.stats as sp2
import streamsculptor
from streamsculptor import potential
from gala.units import UnitSystem
from astropy import units as u
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)

### sim config params
simName = "testRun_yesFDM_m22=2"
N = 320
D = 3
data_drops = 10
cf = .1
L = 10 / np.sqrt(3)
dx = L / N
nf = 1
m22 = np.array([2.01])
rhoDM = 1e7
Mtot = rhoDM * L**3
C = au.G*4*np.pi
Tf = 3500.
initial_drop = 0
T_initial = 0

seed_ = 0

sigma_dm = 216. * au.kms2kpcMyr
n_streams = 64
# print(sigma_dm / au.kms2kpcMyr + 5. / 10000.*Tf)
# print(sigma_dm *Tf)

# returns a random variable in a ball
def randomInBall(Npoints):  
	"""
	returns random variables in a 3 ball

	:Npoints: int, number of points to return
	"""  
	x = np.random.normal(key,Npoints)
	y = np.random.normal(key,Npoints)
	z = np.random.normal(key,Npoints)

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

	# N_stars = len(s.r_stars)
	# s.np = N_stars
	# s.active = np.full(N_stars, False)
	# s.r = np.zeros( (N_stars, 3) )
	# s.v = np.zeros( (N_stars, 3) )
	# r_initial = np.zeros((1,3))
	# r_initial[0] = s.r_prog[0]
	# v_initial = s.v_prog[0]
	# v_mag = np.sqrt(np.sum(np.abs(v_initial)**2))
	# dt = s.t_prog[1] - s.t_prog[0]
	# r_mag = np.sqrt(np.sum(np.abs(r_initial)**2))
	# phi = s.GetPotential(r_initial, 0)
	# # print(s.GetPotential(r_initial, -3500))
	# F = s.GetForceAtPosition(r_initial, 0)
	# F_mag = np.sqrt(np.sum(np.abs(F)**2))
	# print(F*r_initial)
	# v_est1 = np.sqrt(np.sum(-F*r_initial) )
	# v_est2 = np.sqrt(np.abs(phi))
	# print(v_est1 / au.kms2kpcMyr, v_est2 / au.kms2kpcMyr, v_mag / au.kms2kpcMyr)
	# s.T = s.t_prog[1] 
	# print(s.GetCurrentProgVelAndPos()[1])
	# print(v_initial + F*dt)
	# s.StripStar()
	# s.StripStar()
	# print(s.active)
	# print(s.r[s.active])

	# # r_hat = r_initial / r_mag
	# # F_hat = F / F_mag
	# # print(-r_hat, F_hat)

	# assert(0)

	# initialize dynamic variables
	s.set_K()
	Boltzmann_distr(s)

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
	print( s.GetPotential(np.ones((100,3)), 0) )
	# # assert(0)
	# s.GetForceAtPosition(np.ones((3,3)), 0)

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	PlotStuff(np.abs(s.psi[0,N//2])**2)
	s.RunSim()
