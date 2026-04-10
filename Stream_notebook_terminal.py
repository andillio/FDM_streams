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

#Load GD-1
prog_wtoday = jnp.load('GD1_prog/GD1_progenitor.npy', allow_pickle=True).item()
print(prog_wtoday)
pos = prog_wtoday.pos.xyz.to(u.kpc).value
vel = prog_wtoday.vel.d_xyz.to(u.kpc/u.Myr).value
w_today = jnp.hstack([pos,vel])
print(w_today)
## Define a potential
pot_MW = potential.GalaMilkyWayPotential(units=usys)

## Define a stream progenitor (observed today)
#w_today = jnp.array([20.0,0.0,20,.0,.15,.0]) # position units: kpc, velocity units: kpc/Myr

## Stream age
t_age = 3_500 # Myr

## Times to backwards integrate through
ts_back = jnp.linspace(0,-t_age, 1000) #save 1000 points
t_strip = jnp.linspace(-t_age,0, 5000)
Mcluster0 = 1e4 # Msun
Mcluster = jnp.linspace(Mcluster0,0.0,len(t_strip))
init_cond = pot_MW.integrate_orbit(w0=w_today, ts=ts_back,t0=0.0, t1=-t_age,)
print(init_cond)


## Initial condition at t1 = -t_age
IC = init_cond.ys[-1]
print(IC)
print('Initial condition = ' + str(IC))

lead,trail = pot_MW.gen_stream_scan(ts=t_strip, prog_w0=IC, Msat=Mcluster, seed_num=583)
stream = jnp.vstack([lead,trail])
stream = np.asarray(stream)

fig, ax = plt.subplots(1,1)
fig.set_size_inches(6,6)
ax.scatter(stream[:,1], stream[:,2], s=1,rasterized=True,color='k')
ax.set_xlabel('y [kpc]')
ax.set_ylabel('z [kpc]')
#ax.set_xlim(-10,10)
#ax.set_ylim(17,21)
ax.set_aspect('equal')

np.save("stream_final_conditions.npy", stream)
plt.show()

def background_potential(x, y, z, t, potential_object=pot_MW):
    """Return gravitational potential [kpc^2 / Myr^2] at (x, y, z, t). Input units are kpc and Myr. t=0 is today."""
    return potential_object.potential([x, y, z], t)

def get_stream_initial_conditions(t_strip=t_strip, progenitor_IC=IC, progenitor_mass_at_t_strip=Mcluster, potential_object=pot_MW, seed_num=583):
    """Return 6D (x, y, z, vx, vy, vz) initial conditions of each stream star in (leading_arm, trailing_arm).
    Input is (stripping times [from t_start to today=0], progenitor 6D phase space at t_start,
    progenitor mass at stripping times, Potential object, random seed number). Units are kpc and Myr."""
    stream_IC = potential_object.gen_stream_ics(ts=t_strip, prog_w0=progenitor_IC, Msat=progenitor_mass_at_t_strip, seed_num=seed_num)
    stream_IC_leading = jnp.concatenate((stream_IC[0], stream_IC[2]), axis=1)
    stream_IC_trailing = jnp.concatenate((stream_IC[1], stream_IC[3]), axis=1)
    return (stream_IC_leading, stream_IC_trailing)

print(background_potential(20., 0., 0., 0.))

stream_leading, stream_trailing = get_stream_initial_conditions()
# stream = jnp.vstack([lead,trail])
# print(stream_leading.shape)
# fo = pu.FigObj()
# fo.AddPlot(stream_leading[:,0],stream_leading[:,1], ls= '', mk = 'o')
# fo.show()
# stream = jnp.vstack([lead,trail])
# stream = np.asarray(stream)
stream_ICs = pot_MW.gen_stream_ics(ts=t_strip, prog_w0=IC, Msat=Mcluster, seed_num=583)
print(np.sum(stream_ICs[0] - stream_leading[:, :3]), np.sum(stream_ICs[1] - stream_trailing[:, :3]),
      np.sum(stream_ICs[2] - stream_leading[:, 3:]), np.sum(stream_ICs[3] - stream_trailing[:, 3:]))

print(np.asarray(stream_ICs).shape )
stream_ICs = np.asarray(stream_ICs)
# we want from all this stuff
# - prog position
r_prog = init_cond.ys[:,0:3]
r_prog = np.flip(r_prog, axis = 0) # flip time to be in correct order
np.save("r_prog1.npy", r_prog)
# fo = pu.FigObj()
# fo.AddPlot(r_prog[:,0],r_prog[:,1], ls= '', mk = 'o')
# fo.show()
# - prog velocity
v_prog = init_cond.ys[:,3::]
v_prog = np.flip(v_prog, axis = 0) # flip time to be in correct order
np.save("v_prog1.npy", v_prog)
print(v_prog.shape)
# - prog time affine
ts_back = np.flip(ts_back, axis = 0) # flip time to be in correct order
ts_back -= np.min(ts_back)
np.save("t_prog1.npy", ts_back)
print(ts_back)
# - stellar initial positions
N_stars = len(t_strip)
r_stars = np.zeros((N_stars*2, 3))
r_stars[0:N_stars] = stream_ICs[0,:]
r_stars[N_stars::] = stream_ICs[1,:]
np.save("r_stars1.npy", r_stars)
# fo = pu.FigObj()
# fo.AddPlot(stream_ICs[2][:, 1], stream_ICs[2][:, 2], ls= '', mk = 'o')
# fo.show()

# - stellar initial velocities
N_stars = len(t_strip)
v_stars = np.zeros((N_stars*2, 3))
v_stars[0:N_stars] = stream_ICs[2,:]
v_stars[N_stars::] = stream_ICs[3,:]
np.save("v_stars1.npy", v_stars)
# - stellar stripping times
t_strip -= np.min(t_strip)
print(t_strip)
np.save("t_strip1.npy", t_strip)

potential_at_prog = background_potential(
    r_prog[:,0],r_prog[:,1],r_prog[:,2],ts_back)
print(potential_at_prog)

print("completed")