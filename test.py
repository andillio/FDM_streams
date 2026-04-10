# pylint: disable=C,W
import numpy as np
import numpy.linalg as npl
gpu = True
if gpu:
	try:
		import cupy as np
		import cupy.linalg as npl
		import cupy as np
	except:
		pass
import astroUtils as au
import gridUtils as gu
import time 
import sysUtils as su
import sys
sys.path.insert(1, 'Solvers')
import solver
import scipy.stats as sp2

### sim config params
simName = "testRun"
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
np.random.seed(seed_)

sigma_dm = 8. * au.kms2kpcMyr
n_streams = 64


# returns a random variable in a ball
def randomInBall(Npoints):  
	"""
	returns random variables in a 3 ball

	:Npoints: int, number of points to return
	"""  
	x = np.random.normal(0,1,Npoints)
	y = np.random.normal(0,1,Npoints)
	z = np.random.normal(0,1,Npoints)

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


def Boltzmann_distr():
	X,Y,Z = gu.grid((N,N,N),L,gpu=gpu)
	psi = np.zeros((nf,N,N,N)) + 0j
	hbar_ = au.h_tilde(m22)
	stream_velocities = np.zeros((nf, n_streams,3))

	time0 = time.time()

	for j in range(nf):

		v_streams = randomInBall(n_streams)*sigma_dm
		stream_velocities[j,:] = v_streams

		for i in range(n_streams):
		
			v_ = v_streams[i]
			k_ = v_ / hbar_[j]
			S_ = np.rint(k_*L / 2 / np.pi)
			v_ = S_*2*np.pi * hbar_[j] / L
			v_mag = np.sqrt(np.sum(np.abs(v_)**2))
			w_ = 1.#np.exp(-.5*(v_mag/sigma_dm)**2)
			arg = -1j*(v_[0]*X + v_[1]*Y + v_[2]*Z) / hbar_[j]
			phi = np.random.uniform(0,2*np.pi)
			psi[j,:,:,:] += np.sqrt(w_)*np.exp(arg)*np.exp(1j*phi)
			done = i + j*len(v_streams) + 1

			su.PrintTimeUpdate(done, nf*n_streams, time0)

		psi[j,:,:,:] /= np.sqrt(np.sum(np.abs(psi[j,:,:,:])**2)*dx**3)
		psi[j,:,:,:] *= np.sqrt(Mtot)

		return psi


def SetICs():
	s = solver.Solver()

	# sim params
	s.simName = simName
	s.N = N 
	s.np = 0
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

	# initialize dynamic variables
	s.psi = Boltzmann_distr()
	s.r = np.array([])
	s.v = np.array([])
	s.set_K()
	s.sigma = sigma_dm * np.sqrt(8/np.pi)

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	s.RunSim()
