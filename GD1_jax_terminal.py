from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
mpl.rcParams['text.usetex'] = False
import numpy as np

# gala
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Must be set before importing jax
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import dimensionless, galactic, UnitSystem
import time

import jax
import jax.numpy as jnp

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

import plotUtils as pu

from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)

import sys, importlib
importlib.reload(JaxStreams)




usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)

t0 = 0  # start time (present)
t1 = -2000  # end time (-2 Gyr)

def IntegrateOrbit():
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

	n_keep_orbit = 1000
	ts = jnp.linspace(t0, t1,n_keep_orbit)
	progenitor_backwards = pot.orbit_integrator_run_notdense(wf,0,t1,ts,None)[:-1,:]

	return progenitor_backwards, pot


def PlotOrbit(progenitor_backwards):
	fo = pu.FigObj()

	fo.AddPlot(progenitor_backwards[:,0],progenitor_backwards[:,2])
	fo.AddLine(progenitor_backwards[0, 0], progenitor_backwards[0, 2], color = 'k',
	 ls = '', mk = 'o', label = 'Now')
	fo.AddLine(progenitor_backwards[-1, 0], progenitor_backwards[-1, 2], color = 'r',
	 ls = '', mk = 'o', label = r'$-2 \, \mathrm{Gyr}$')

	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$z \, [\mathrm{kpc}]$')

	fo.SetTitle('Integrated GD-1 orbit (Galactocentric)')
	fo.legend()
	fo.show()


def PlotGD1Stream(progenitor_backwards, stream):
	fo = pu.FigObj()

	fo.AddPlot(progenitor_backwards[:,0],progenitor_backwards[:,2])
	fo.AddLine(stream[:,0],stream[:,2], color = 'r',
	 ls = '', mk = '.', label = r'Mock stream')

	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$z \, [\mathrm{kpc}]$')

	fo.SetTitle('Generated GD-1 stream along orbit (Galactocentric)')
	fo.legend()
	fo.show()


def GenerateStream(pot, progenitor_backwards):
	# Now generate a mock stream along the above orbit
	seed_num = 4030
	M_sat = .5e4 #progenitor mass
	t_strip = jnp.linspace(t1,t0,1_000) 

	start = time.time()
	lead_arm, trail_arm = pot.gen_stream_vmapped_notdense(
		t_strip, progenitor_backwards[-1], M_sat,seed_num,False)
	end = time.time()
	print(end-start)

	stream = jnp.vstack([lead_arm,trail_arm])[:,0,:] # x, y, z, vx, vy, vz
	return stream


# Define function to convert from RA, Dec to stream coordinates phi1, phi2
@jax.jit
def icrs_to_gd1(ra_rad, dec_rad):
    R = jnp.array(
        [
            [-0.4776303088, -0.1738432154, 0.8611897727],
            [0.510844589, -0.8524449229, 0.111245042],
            [0.7147776536, 0.4930681392, 0.4959603976],
        ]
    ) # R is the rotation matrix from Koposov et al. (2010)


    icrs_vec = jnp.vstack([jnp.cos(ra_rad)*jnp.cos(dec_rad),
                           jnp.sin(ra_rad)*jnp.cos(dec_rad),
                           jnp.sin(dec_rad)]).T 
    # spherical coords to geocentric equatorial coords X, Y, Z

    stream_frame_vec = jnp.einsum('ij,kj->ki',R,icrs_vec) # matrix multiplication?
    # stream_frame_vec = jnp.dot(R,icrs_vec).T
    
    phi1 = jnp.arctan2(stream_frame_vec[:,1],stream_frame_vec[:,0])*(180/jnp.pi)
    phi2 = jnp.arcsin(stream_frame_vec[:,2])*(180/jnp.pi)

    
    return phi1, phi2


def CreateICRS_coords(stream):
	# Create SkyCoord object in Galactocentric coordinates of generated stars in stream
	stream_xyz = coord.SkyCoord(x=stream[:,0], y=stream[:,1], z=stream[:,2], unit='kpc', 
	                            frame = 'galactocentric')
	# Change to ICRS coordinates
	stream_icrs = stream_xyz.transform_to(coord.ICRS).data
	gd1_ras = stream_icrs.lon.value  # in rad
	gd1_decs = stream_icrs.lat.value  # in rad

	return gd1_ras, gd1_decs


def PlotTranformedStream(gd1_ras, gd1_decs):
	fo = pu.FigObj()

	phi1, phi2 = icrs_to_gd1(gd1_ras, gd1_decs)
	prog1, prog2 = icrs_to_gd1(np.deg2rad(gd1_c.ra.value), np.deg2rad(gd1_c.dec.value))  # RA and Dec of progenitor
	fo.AddPlot(phi1, phi2, color = 'r',
	 ls = '', mk = '.', label = r'Mock stream')
	fo.AddPlot(prog1, prog2, color = 'k',
	 ls = '', mk = '*', label = r'Mock stream')


if __name__ == "__main__":
	progenitor_backwards, pot = IntegrateOrbit()
	# PlotOrbit(progenitor_backwards)
	stream = GenerateStream(pot, progenitor_backwards)
	# PlotGD1Stream(progenitor_backwards, stream)
	gd1_ras, gd1_decs = CreateICRS_coords(stream)
	PlotTranformedStream(gd1_ras, gd1_decs)