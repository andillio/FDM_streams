# pylint: disable=C,W
import numpy as np_
import os
import time
import astroUtils as au
import sysUtils as su
import gridUtils as gu
import mathUtils as mu
import types
import baseSolver as BS
CUPY_IMPORTED = True
import warnings as warn 
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False
import cupy as cp
import jax.dlpack
import jax.numpy as jnp

class Solver():

	def __init__(self):
		### simulation paramters
		self.simName = "" # string, folder name for sim data
		self.N = 1 # int, sim resolution
		self.np = 0 # int, num particles
		self.mp = 0 # array[np], array of particle masses
		self.data_drops = 0 # int, number of data drops (past initial drop)
		self.initial_drop = 0 # what drop should I start counting from
		self.T_initial = 0 # what time should I start the sim from
		self.cf = 1.0 # float, courant factor
		self.gpu = False # bool, run on gpu
		self.dt_max = 1.
		self.freezeDensity = False
		self.psiSelfGrav = False
		self.integrateBackwards = False
		self.shouldStripStars = True

		### physics parameter
		self.L = 1. # float, box length
		self.dx = 1. # float, pixel size
		self.nf = 1 # int, number of fields
		self.D = 3 # int, number of spatial dimensions
		self.C = 1. # float, poisson's constant
		self.Tf = 1. # float, final time
		self.m22 = 1 # float, particle mass [1e-22 eV/c^2]
		self.hbar_ = 1. # float, hbar / m [kpc^2 / Myr] 
		self.padded = False

		### dynamic variables
		self.psi = None
		self.K = None # array-like, [N^3] kinetic update operator argument, kx x kx x kx
		self.r = None # array-like, [np, D] positions of particles
		self.v = None # array-like, [np, D] velocities of particles
		self.active = None # array-like, [np] is this star currently simulated?
		self.dt = 1. # float, timestep
		self.dt_factor = 1 + 0j # factor to make things imaginary
		self.sigma = 0. # temperature of the dark matter
		self.v_bulk = None 
		self.T = 0
		self.T_ref = 0

		### input params
		self.r_prog = None # (1,3) current prog position
		self.v_prog = None # (1,3) current prog velocity
		self.r_stars = None 
		self.v_stars = None 
		self.v_frame = None 
		self.t_stars = None
		self.strip_index = 0

		### diagnostics
		self.times = []
		self.percents = []
		self.counter = 0
		self.timesMax = 100

		self.extras = {} # dictionary to add to toml


	def SetParams(self,**kwargs):
		# simulation params
		if kwargs.get("simName"):
			self.simName = kwargs['simName']
		if kwargs.get("N"):
			self.N = kwargs['N']
		if kwargs.get("np"):
			self.np = kwargs['np']
		if kwargs.get('mp'):
			self.mp = kwargs['mp']
		if kwargs.get('data_drops'):
			self.data_drops = kwargs['data_drops']
		if kwargs.get('cf'):
			self.cf = kwargs['cf']
		if kwargs.get('freezeDensity'):
			self.freezeDensity = kwargs['freezeDensity']
		if kwargs.get('psiSelfGrav'):
			self.psiSelfGrav = kwargs['psiSelfGrav']

		# physics params
		if kwargs.get('L'):
			self.L = kwargs['L']
		if kwargs.get('dx'):
			self.dx = kwargs['dx']
		if kwargs.get('nf'):
			self.nf = kwargs['nf']
		if kwargs.get('D'):
			self.D = kwargs['D']
		if kwargs.get('C'):
			self.C = kwargs['C']
		if kwargs.get('Tf'):
			self.Tf = kwargs['Tf']
		if kwargs.get('m22'):
			self.m22 = kwargs['m22']
		if kwargs.get('hbar_'):
			self.hbar_ = kwargs['hbar_']


	def OutputICs(self):
		"""
		outputs the initial conditions
		"""
		if not(os.path.isdir(f"Data/{self.simName}/r")) and self.np != None and self.np > 0:
			os.mkdir(f"Data/{self.simName}/r")
		if not(os.path.isdir(f"Data/{self.simName}/v")) and self.np != None and self.np > 0:
			os.mkdir(f"Data/{self.simName}/v")
		if not(os.path.isdir(f"Data/{self.simName}/psi")):
			os.mkdir(f"Data/{self.simName}/psi")
		self.DataDrop(0)


	def DataDrop(self, i):
		"""
		output the current status of the dynamical variables
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		if self.np != None and self.np > 0:
			np.save("Data/" + self.simName + f"/r/drop{i + self.initial_drop}.npy", self.r)
			np.save("Data/" + self.simName + f"/v/drop{i + self.initial_drop}.npy", self.v)
		np.save("Data/" + self.simName + f"/psi/drop{i+ self.initial_drop}.npy", self.psi)

		self.r_prog_save[i + self.initial_drop] = self.r_prog[0]
		self.v_prog_save[i + self.initial_drop] = self.v_prog[0]
		np.save("Data/" + self.simName + f"/r_prog.npy", self.r_prog_save)
		np.save("Data/" + self.simName + f"/v_prog.npy", self.v_prog_save)


	def OutputToml(self):
		"""
		outputs toml with simulation parameters
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		mp = self.mp 
		if (isinstance(mp, float) or isinstance(mp,int)):
			mp = np.ones(self.np)*self.mp
		mp = np.mean(mp)

		text = f'''
		# all units in kpc, Msolar, Myr
		[physics]
		Tf              	        = {self.Tf + self.T_initial} # float, final sim time
		L                           = {self.L} # float, box length
		C 							= {self.C} # float, poisson's constant
		D 							= {self.D} # int, number of spatial dimensions
		m22 						= {np.array2string(self.m22, separator=', ')} # m_field / 1e-22 eV
		hbar_ 						= {np.array2string(self.hbar_, separator=', ')} # hbar / m_field
		dx 							= {self.dx} # float, box physical resolution
		nf 							= {self.nf} # int, number of ultralight fields

		[simulation]
		N                           = {self.N} # int, grid size
		np                			= {self.np} # int, number of simulation particles
		data_drops            	    = {self.data_drops + self.initial_drop} # int, number of data drops
		cf                         	= {self.cf} # float, timestep courant factor
		freezeDensity 				= {str(self.freezeDensity).lower()} # bool, field does not have time udpate
		psiSelfGrav 				= {str(self.psiSelfGrav).lower()} # bool, field feels own gravity
		'''

		f = open(f"Data/{self.simName}/meta.toml", "w")
		f.write(text)
		f.close()

		if len(self.extras) > 0:
			extras = {}
			extras['extras'] = self.extras
			su.AddLines2Toml(extras, f"Data/{self.simName}/meta.toml")

	def InitializeFiles(self):
		"""
		makes the data directory and outputs the toml file
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		if self.simName == None:
			raise Exception("simName has not been set.\n"+\
				"set simName before initializing files.")

		if not(os.path.isdir(f"Data/{self.simName}")):
			os.mkdir(f"Data/{self.simName}")

		self.r_prog_save = np.zeros((self.data_drops+1,3))
		self.v_prog_save = np.zeros((self.data_drops+1,3))	

		self.OutputToml()
		self.OutputICs()

	def set_K(self, K = []):
		"""
		set the spectral gird

		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op \n
			default: use N,L, and D to figure K out
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(K) > 0:
			self.K = K 
		else:
			if self.N != None and self.L != None and self.D != None:
				dx = self.L/self.N
				kx = 2*np.pi*np.fft.fftfreq(self.N,d = dx)
				ones = np.ones(self.N)

				self.K = self.get_K(self.N, self.L)
	def get_K(self, N, L):
		"""
		calculate the spectral grid

		:N: int, the grid resolution
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		dx = L/N
		kx = 2*np.pi*np.fft.fftfreq(N, d = dx)
		ones = np.ones(N)

		if self.D == 1:
			return kx**2
		elif self.D == 2:
			K = np.einsum("i,j->ij", kx**2, ones)
			K += np.einsum("i,j->ij", ones, kx**2)
			return K
		elif self.D == 3:
			K = np.einsum("i,j,k->ijk", kx**2, ones, ones)
			K += np.einsum("i,j,k->ijk", ones, kx**2, ones)
			K += np.einsum("i,j,k->ijk", ones, ones, kx**2)
			return K

	def GetFFt(self, psi, Forward = True, NoFieldDimension = False):
		"""
		calculate the fft of psi

		:psi: array-like, [nf, N^D], the fields
		:Forward: bool, forward or backward fft, default: True
		:NoFieldDimension: bool, if True then psi has dim[N^D]

		:return: array-like, fft of psi on all axes except 0th
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if self.D == None:
			self.D = len(psi.shape) - 1

		a = NoFieldDimension

		if Forward:
			if self.D == 1:
				return np.fft.fft(psi, axis = 1 - a)
			elif self.D == 2:
				return np.fft.fft2(psi, axes = (1-a,2-a))
			elif self.D == 3:
				return np.fft.fftn(psi, axes = (1-a,2-a,3-a))
		else:
			if self.D == 1:
				return np.fft.ifft(psi, axis = 1-a)
			elif self.D == 2:
				return np.fft.ifft2(psi, axes = (1-a,2-a))
			elif self.D == 3:
				return np.fft.ifftn(psi, axes = (1-a,2-a,3-a))

	# def MakePeriodic(self):
	# 	if len(self.r) > 0:
	# 		self.r += self.L/2.
	# 		self.r %= self.L
	# 		self.r -= self.L/2.

	def GetPeriodicR(self, r_notPeriodic):
		r = r_notPeriodic + self.L / 2.
		r %= self.L 
		r -= self.L / 2.
		return r

	def compute_phi(self):
		"""
		compute the potential

		:include_particles: bool, include particle density in phi calc
		:include_external: bool, include effect or external forces
	
		:return: array-like, [N^D]
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		rval = np.sum(np.abs(self.psi)**2, axis = 0)

		rval = self.GetFFt(rval, NoFieldDimension=True)

		K = self.K
		rval = -1*self.C*rval / K
		if self.D == 3:
			rval[0,0,0] = 0.0
		elif self.D ==2:
			rval[0,0] = 0.
		elif self.D == 1:
			rval[0] = 0

		rval = self.GetFFt(rval, Forward = False, NoFieldDimension=True)
		rval = rval.real

		return rval


	def get_dt(self, T_remaining, Vmax = None):
		"""
		calculate the timestep

		:T_remaining: float, the time remaining until the next data drop, \n
					default: do not consider this condition
		:Vmax: float, max value of the potential, default: calculate Vmax
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		dt_base = T_remaining
		if not(self.np == None or self.np == 0):
			dt_all = np.zeros(3)
			dt_all[0] = self.cf * self.dx / np.max(np.abs(self.v))
			dt_all[1] = T_remaining
			dt_all[2] = self.dt_max
			dt_base = np.min(dt_all)

		if self.freezeDensity:
			return dt_base


		### kinetic time check
		k_max = np.sqrt(self.D) * np.pi / self.dx
		delta_k = 2 * k_max / self.N

		dt_kinetic = self.cf * 4 *np.pi / k_max**2 / np.max(self.hbar_)
		
		### potential time check
		if Vmax == None:
			phi = self.compute_phi()
			phi -= np.mean(phi)
			Vmax = np.max(np.abs(phi))
		dt_potential = self.cf * 2 * np.pi * np.min(self.hbar_) / Vmax

		dt_array = np.zeros(3)
		dt_array[0] = dt_base
		dt_array[1] = dt_kinetic
		dt_array[2] = dt_potential
		return np.min(dt_array)

	def RunSim(self):
		"""
		run the simulation
		"""
		self.InitializeFiles()
		print("\nrunning simulation " + self.simName + "..." )
		time0 = time.time()
		tNext = float(self.Tf)/self.data_drops

		# strip first star 
		tNextStar = 0
		if self.shouldStripStars:
			self.strip_index = 0
			self.StripStar()
			tNextStar = self.t_stars[self.strip_index]

		drop = 1
		self.T = 0.
		Vmax = None

		# self.MakePeriodic()

		while(self.T < self.Tf):

			T_remaining_drop = tNext - self.T
			T_remaining_star = tNextStar - self.T

			T_remaining = T_remaining_drop
			if T_remaining_star < T_remaining and self.shouldStripStars:
				T_remaining = T_remaining_star

			dt = self.get_dt(T_remaining, Vmax=Vmax)
			Vmax = self.Update(dt)
			self.T += dt 
			self.PrintDiagnostics(self.T, time0)
			if T_remaining_drop <= 0:
				# su.PrintTimeUpdate(drop,self.data_drops,time0)
				self.DataDrop(drop)
				drop += 1
				tNext = float(self.Tf*drop)/self.data_drops
			if T_remaining_star <= 0 and self.shouldStripStars:
				self.StripStar()
				if self.strip_index < len(self.t_stars):
					tNextStar = self.t_stars[self.strip_index]
				else:
					tNextStar = self.Tf + 1

		if (drop == self.data_drops):
			self.DataDrop(drop)

		su.PrintCompletedTime(time0, "simulation")

	def StripStar(self):
		# we need to get the interp the progenitor location and velocity

		r_prog_current, v_prog_current = self.GetCurrentProgVelAndPos()

		# then spawn a star in the frame of ref of the prog
		r_new_star = self.r_stars[self.strip_index] - r_prog_current
		v_new_star = self.v_stars[self.strip_index] - v_prog_current
		# then add that star to the simulated list

		self.r[self.strip_index] = r_new_star
		self.v[self.strip_index] = v_new_star
		self.active[self.strip_index] = True

		N_stars_per_arm = len(self.r_stars) // 2
		# then spawn a star in the frame of ref of the prog
		r_new_star = self.r_stars[self.strip_index + N_stars_per_arm] - r_prog_current
		v_new_star = self.v_stars[self.strip_index + N_stars_per_arm] - v_prog_current
		# then add that star to the simulated list

		self.r[self.strip_index + N_stars_per_arm] = r_new_star
		self.v[self.strip_index + N_stars_per_arm] = v_new_star
		self.active[self.strip_index + N_stars_per_arm] = True
		self.strip_index += 1


	def GetCurrentProgVelAndPos(self):
		dt_prog = self.t_prog_implicit[1] - self.t_prog_implicit[0]
		left_index = int( self.T / dt_prog )
		right_index = left_index + 1
		fraction_right = (self.T - left_index * dt_prog) / dt_prog
		fraction_left = 1. - fraction_right

		r_prog_current = self.r_prog_implicit[left_index]*fraction_left \
		    + self.r_prog_implicit[right_index]*fraction_right
		v_prog_current = self.v_prog_implicit[left_index]*fraction_left \
		    + self.v_prog_implicit[right_index]*fraction_right

		return r_prog_current, v_prog_current


	def BaseDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started

		:return: string, diagnostic info
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		T_remaining = self.Tf - T
		elapsedTimeTotal = time.time() - time0
		portionDone = T / self.Tf
		portionRemaining = 1 - portionDone

		if self.counter == 0:
			self.times = np.zeros(self.timesMax)
			self.percents = np.zeros(self.timesMax)

		ind_ = self.counter%self.timesMax
		self.times[ind_] = time.time()
		self.percents[ind_] = portionDone

		elapsedTimeRecently = elapsedTimeTotal
		portionDoneRecently = portionDone

		if self.counter == self.timesMax:
			elapsedTimeRecently = \
				(np.max(self.times) - np.min(self.times)) / (len(self.times) - 1)
			portionDoneRecently = \
				(np.max(self.percents) - np.min(self.percents)) / (len(self.percents) - 1)

		self.counter += 1

		string_done = "%.2f done "%(portionDone) 
		string_done += "in %i hrs, %i mins, %i s."%su.hms(elapsedTimeTotal)

		time_estimate = portionRemaining*elapsedTimeRecently / portionDoneRecently
		string_todo = f" eta: %i hrs, %i mins, %i s."%su.hms(time_estimate)
		return string_done + string_todo

	def PrintDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started
		"""
		str_ = self.BaseDiagnostics(T,time0)
		str_ += " %.2f alias fraction"%(self.rho_alias)
		su.repeat_print(str_)

	def Update(self, dt):
		"""
		updates the dynamic variables using a drift-kick-drift scheme

		:dt: float, the timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if self.integrateBackwards:
			dt = np.abs(dt)*-1
		self.Drift(dt/2.) # alter velocity phases (ie positions) based on velocities
		self.AlterTemp() # alter velocity amplitudes based on position
		self.AlterAmp()
		Vmax = self.Kick(dt) # alter position phases (ie velocity) based on position
		self.Drift(dt/2.) # alter velocity phases based velocities

		return Vmax

	def AlterAmp(self):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		r_prog = self.r_prog
		R = np.sqrt(np.sum(np.abs(r_prog)**2))

		rs = 20.
		rho_s = 8.5e6 
		Mtot = au.NFW(R, rs, rho_s) * self.L**3
		for i in range(self.nf):		
			self.psi[i] /= np.sqrt(np.sum(np.abs(self.psi[i])**2)*self.dx**3)
			self.psi[i,:,:,:] *= np.sqrt(Mtot)


	def AlterTemp(self):
		"""

		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		# update sigma
		# r_prog_current, v_prog_current = self.GetCurrentProgVelAndPos()
		r_prog = self.r_prog
		F = self.GetForceAtPosition(r_prog, self.T)
		self.sigma = np.sqrt( np.abs( np.sum(-F*r_prog)) )

		# get initial norms
		norm = np.zeros(self.nf)
		for i in range(self.nf):
			norm[i] = np.sum(np.abs(self.psi[i])**2)*self.dx**3

		# change the temperature
		self.psi = self.GetFFt(self.psi) # get the psi on k
		self.psi = np.exp(-np.einsum("i,jkl->ijkl",self.hbar_**2, self.K) /\
		 2. / self.sigma**2) * np.exp(np.angle(self.psi)*1j)
		self.psi = self.GetFFt(self.psi, Forward=False)

		# renorm everything
		for i in range(self.nf):		
			self.psi[i] /= np.sqrt(np.sum(np.abs(self.psi[i])**2)*self.dx**3)
			self.psi[i,:,:,:] *= np.sqrt(norm[i])



	def Drift(self, dt):
		"""
		update the dynamic variable positions

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		dt = self.dt_factor * (dt + 0j)

		# if there is a bulk velocity then add it to the field
		if not(self.v_bulk is None):
			v_bulk = np.einsum("i,j->ij", 1./self.hbar_, self.v_bulk)
			self.psi *= np.exp(-1j*np.einsum("ij,jkln->ikln", v_bulk,
				np.array(gu.grid((self.N,self.N, self.N), L = self.L, gpu = self.gpu))) )

		# update the field
		if not(self.freezeDensity):
			k2max = (np.pi*self.N/self.L)**2

			self.psi = self.GetFFt(self.psi)

			if self.D == 3:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,jkl->ijkl",self.hbar_, self.K)/(2.))
			elif self.D == 2:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,jk->ijk",self.hbar_, self.K)/(2.))
			elif self.D == 1:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,j->ij",self.hbar_, self.K)/(2.))


			rhoOverThresh = 0
			if self.nf == 1:
				if self.D == 3:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:,:,:] > (k2max*.9)])**2)
				elif self.D == 2:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:,:] > (k2max*.9)])**2)
				elif self.D == 1:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:] > (k2max*.9)])**2)
			
			rhoToT = np.sum(np.abs(self.psi)**2)
			self.rho_alias = rhoOverThresh / rhoToT

			self.psi = self.GetFFt(self.psi, Forward=False)

		# if there is a bulk velocity then add it to the field
		if not(self.v_bulk is None):
			v_bulk = np.einsum("i,j->ij", 1./self.hbar_, self.v_bulk)
			self.psi *= np.exp(1j*np.einsum("ij,jkln->ikln", v_bulk,
				np.array(gu.grid((self.N,self.N, self.N), L = self.L, gpu = self.gpu))) )


		dt = np.real(dt)
		if len(self.r[self.active]) > 0 and len(self.v[self.active]) > 0:
			self.r[self.active] += self.v[self.active] * dt
			self.r_prog += self.v_prog * dt
			# self.MakePeriodic()


	def ComputeAcc(self, phi, r):
		"""
		compute the acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(np.shape(r))

		if len(phi) > 0:
			if self.D == 1:
				acc = self.ComputeAcc1D(phi)
			elif self.D == 2:
				acc = self.ComputeAcc2D(phi)
			elif self.D == 3:
				acc = self.ComputeAcc3D(phi,r)		
		return acc



	def ComputeAcc3D(self, phi, r):
		"""
		compute the 3D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.GetPeriodicR(r)
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		for i in range(self.D):
			axis_ = i 
			gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded= self.padded)

			# i,j,k
			f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
			f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
			f3 = 1 - np.abs(x[ijk[:,2]] - r[:,2])/dx
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j,k
			f1 = 1. - f1 # dx
			ijk[:,0] += 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j+1,k
			f2 = 1. - f2 # tx
			ijk[:,1] += 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j+1,k
			f1 = 1. - f1
			ijk[:,0] -= 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j+1,k+1
			f3 = 1. - f3
			ijk[:,2] += 1
			ijk[:,2] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j,k+1
			f2 = 1. - f2
			ijk[:,1] -= 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j,k+1
			f1 = 1. - f1
			ijk[:,0] += 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j+1,k+1
			f2 = 1. - f2
			ijk[:,1] += 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			ijk[:,0] -= 1
			ijk[:,1] -= 1
			ijk[:,2] -= 1

		return -1*acc

	def ComputeAcc2D(self, phi):
		"""
		compute the 2D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.r[self.active]
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		for i in range(self.D):
			axis_ = i 
			gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded=self.padded)

			# i,j
			f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
			f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i+1,j
			f1 = 1. - f1 # dx
			ijk[:,0] += 1
			ijk[:,0] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i+1,j+1
			f2 = 1. - f2 # tx
			ijk[:,1] += 1
			ijk[:,1] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i,j+1
			f1 = 1. - f1
			ijk[:,0] -= 1
			ijk[:,0] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			ijk[:,0] -= 1
			ijk[:,1] -= 1

		return -1*acc


	def ComputeAcc1D(self, phi):
		"""
		compute the 1D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.r[self.active]
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		axis_ = 0 
		gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded=self.padded)

		# i
		f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
		acc[:,0] += gradPhi[ijk[:,0],ijk[:,1]]*f1

		# i+1
		f1 = 1. - f1 # dx
		ijk[:,0] += 1
		ijk[:,0] %= N
		acc[:,0] += gradPhi[ijk[:,0],ijk[:,1]]*f1

		return -1*acc


	def Kick(self, dt):
		"""
		update the dynamic variable momenta

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		
		particles_need_kick = not(self.np is None) and self.np > 0
		fields_need_kick = not(self.freezeDensity) and not(self.psiSelfGrav)
		if not(particles_need_kick) and not(fields_need_kick):
			return 0

		phi = self.compute_phi()

		dt = self.dt_factor * (dt + 0j)

		if not(self.freezeDensity) and self.psiSelfGrav:
			if self.D == 3:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,jkl->ijkl",1./self.hbar_, phi))
			elif self.D == 2:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,jk->ijk",1./self.hbar_, phi))
			elif self.D == 1:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,j->ij",1./self.hbar_, phi))
		if not(self.np is None) and self.np > 0:
			acc = self.ComputeAcc(phi, self.r[self.active]) # force from field
			acc_prog = self.GetForceAtPosition(self.r_prog
				, float(self.T) ) # force from external
			
			acc[:] -= acc_prog[0]
			acc += self.GetForceAtPosition(self.r[self.active] + self.r_prog[0]
				, float(self.T) ) # relative external force compared to prog ref frame
			
			acc_prog += self.ComputeAcc(phi, self.r_prog) # force from field
			self.v[self.active] += acc*dt.real
			self.v_prog += acc_prog*dt.real

		Vmax = np.max(np.abs(phi - np.mean(phi)))

		return Vmax

	def GetFieldDensity(self):
		'''
		returns field density
		'''
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		return np.sum(np.abs(self.psi)**2, axis = 0)


	def GetPotential(self, r, T):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		jax_array = jax.dlpack.from_dlpack(r)
		return cp.from_dlpack( self.GetPotential_jax(jax_array, float(T)) )

	@jax.jit(static_argnums=0)
	def GetPotential_jax(self,r,T):
	    return jax.vmap(lambda pos: self.pot_MW.potential(pos, T))(r)


	def GetForceAtPosition(self, r, T):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		force = np.zeros(np.shape(r))

		if self.integrateBackwards:
			T = self.T_ref - T

		for i in range(3):
			displacement = np.zeros(3)
			displacement[i] = self.dx
			# calculates the negative gradient
			force[:,i] += (-1./12./self.dx)*self.GetPotential(r - 2*displacement, T)
			force[:,i] += (8./12./self.dx)*self.GetPotential(r - displacement, T)
			force[:,i] += (-8./12./self.dx)*self.GetPotential(r + displacement, T)
			force[:,i] += (1./12./self.dx)*self.GetPotential(r + 2*displacement, T)

		return force
		