from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use('science')
mpl.rcParams['text.usetex'] = False
import numpy as np

# gala
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import dimensionless, galactic, UnitSystem
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax import config
# from jax.config import config
config.update("jax_enable_x64", True)

# from unxt import Quantity

import jax.random as random 
from matplotlib.patches import Ellipse
import jax.scipy.stats as statsjax

usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint

import JaxStreams_CustomForwardMode_MassRadiusImpact as JaxStreams
# from jax.lib import xla_bridge
print(jax.extend.backend.get_backend())

import sys, importlib
importlib.reload(JaxStreams)


# Setting up a global potential
params_global_potential = {'m_disk':5.0e10, 'a_disk': 3.0, 'b_disk': 0.25, 'm_NFW': 1.0e12,
                          'r_s_NFW': 15.0}

pot_disk = JaxStreams.MiyamotoNagaiDisk(m=params_global_potential['m_disk'], a=params_global_potential['a_disk'],
                                       b=params_global_potential['b_disk'],units=usys)
pot_NFW = JaxStreams.NFWPotential(m=params_global_potential['m_NFW'], r_s=params_global_potential['r_s_NFW'],units=usys)

## Combine potentials in a list structure
potential_list = [pot_disk,pot_NFW]
pot = JaxStreams.Potential_Combine(potential_list=potential_list,units=usys)

# Present day location of GD-1 progenitor from Webb & Bovy (2019) https://arxiv.org/pdf/1811.07022
gd1_c = coord.SkyCoord(ra=148.9363998668805*u.degree, 
                        dec=36.15980426805254*u.degree,
                        distance=7.555339165941959*u.kpc,
                        pm_ra_cosdec=-5.332929760383195*u.mas/u.yr, 
                        pm_dec=-12.198914465325117*u.mas/u.yr,
                        radial_velocity=6.944006091929623*u.km/u.s,
                        frame='icrs')
# pm_ra_cosdec is proper motion scaled by the cos of the dec, which accounts for the fact that a change in longitude (ra) 
# has different physical length at different latitudes (dec)
rep = gd1_c.transform_to(coord.Galactocentric).data
gd1_w0 = gd.PhaseSpacePosition(rep) # positions and conjugate momenta (velocities) of GD-1 coords

# current (and final) position of GD-1 progenitor in a jax array
wf = jnp.hstack([gd1_w0.pos.xyz.T.value,gd1_w0.vel.d_xyz.to(u.kpc/u.Myr).value]) # x, y, z, vx, vy, vz

# Integrate orbit of progenitor back 2 Gyr
t0 = 0  # start time (present)
t1 = -2000  # end time (-2 Gyr)
n_keep_orbit = 1000
ts = jnp.linspace(t0, t1,n_keep_orbit)
progenitor_backwards = pot.orbit_integrator_run_notdense(wf,0,t1,ts,None)[:-1,:]
print(progenitor_backwards.shape)

# save the orbital information
# shift the time to 0
t_shifted = np.flip(ts - np.min(ts))
r = np.flip(progenitor_backwards[:,0:3], axis = 0)
v = np.flip(progenitor_backwards[:,3:6], axis = 0)

# np.save("GD1_progenitor.npy", progenitor_backwards)
np.save("t_prog.npy",t_shifted)
np.save("r_prog.npy",r)
np.save("v_prog.npy",v)


# # Plot the integrated orbit
# plt.figure(figsize = (7, 5))
# plt.plot(progenitor_backwards[:,0],progenitor_backwards[:,2]) # plotting x against z?
# plt.scatter(progenitor_backwards[0, 0], progenitor_backwards[0, 2], c = 'black', label = 'Now', zorder = 100)
# plt.scatter(progenitor_backwards[-1, 0], progenitor_backwards[-1, 2], c = 'r', label = '-2 Gyr', zorder = 100)
# plt.xlabel('x [kpc]')
# plt.ylabel('z [kpc]')
# plt.title('Integrated GD-1 orbit (Galactocentric)')
# plt.legend()
# plt.show()