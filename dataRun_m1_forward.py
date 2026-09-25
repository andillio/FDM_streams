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
# import solverExplicitProg as solver
import solverRelease as solver
import scipy.stats as sp2
import streamsculptor
from streamsculptor import potential
from gala.units import UnitSystem
from astropy import units as u
from astropy.constants import G
import numpy as np_
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
usys.G = G.to(u.kpc**3 / (u.Msun * u.Myr**2)).value

### sim config params
simName = "dataRun_m22=1_forward_run1"
refSim = "dataRun_m22=1_back_run1"
N = 256
D = 3
data_drops = 20
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

sigma_dm = 216. * au.kms2kpcMyr
n_streams = 64


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

	s.set_K()

	N_stars = int(1e4)
	s.np = N_stars
	
	s.psi = np.load(f"Data/{refSim}/psi/drop100.npy")
	s.t_stars = np.linspace(0,3500,5000)
	s.r_prog = np.zeros((1,3))
	s.v_prog = np.zeros((1,3))
	s.r_prog[0] = np.load(f"Data/{refSim}/r_prog.npy")[-1]
	s.v_prog[0] = np.load(f"Data/{refSim}/v_prog.npy")[-1]
	
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
