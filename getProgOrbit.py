from gala.units import UnitSystem
from astropy import units as u
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
import jax.numpy as jnp
import plotUtils as pu
import diffrax

import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import streamsculptor
from streamsculptor import potential

t_age = 100 # Myr
t0 = 0
ts_back = jnp.linspace(t0,-t_age, 1000) #save 1000 points

prog_wtoday = jnp.load('GD1_prog/GD1_progenitor.npy', allow_pickle=True).item()
pos = prog_wtoday.pos.xyz.to(u.kpc).value
vel = prog_wtoday.vel.d_xyz.to(u.kpc/u.Myr).value
w_today = jnp.hstack([pos,vel])
pot_MW = potential.GalaMilkyWayPotential(units=usys)
init_cond = pot_MW.integrate_orbit(w0=w_today, ts=ts_back,t0=0, t1=-t_age,)

r_prog = init_cond.ys[:,0:3]
r_prog = np.flip(r_prog, axis = 0) # flip time to be in correct order
v_prog = init_cond.ys[:,3::]

fo = pu.FigObj()
fo.AddPlot(r_prog[:, 1], r_prog[:, 2])

t_age = 100 # Myr
t0 = 0
ts_back2 = jnp.linspace(t0, t_age, 1000) #save 1000 points

prog_wtoday = jnp.load('GD1_prog/GD1_progenitor.npy', allow_pickle=True).item()
pos = prog_wtoday.pos.xyz.to(u.kpc).value
vel = prog_wtoday.vel.d_xyz.to(u.kpc/u.Myr).value
w_today = jnp.hstack([pos,vel])
pot_MW = potential.GalaMilkyWayPotential(units=usys)
init_cond = pot_MW.integrate_orbit(w0=w_today, ts=ts_back2,t0=0, t1= t_age,)

r_prog2 = init_cond.ys[:,0:3]
# r_prog2 = np.flip(r_prog, axis = 0) # flip time to be in correct order
v_prog2 = init_cond.ys[:,3::]

fo.AddLine(r_prog2[:, 1], r_prog2[:, 2], color = 'r')


r_prog = np.concatenate( (r_prog, r_prog2) )
v_prog = np.concatenate( (v_prog, v_prog2) )
ts_back = np.concatenate( (ts_back, ts_back2) )

np.save("t_prog_orbit.npy", ts_back)
np.save("v_prog_orbit.npy", v_prog)
np.save("r_prog_orbit.npy", r_prog)


fo.show()