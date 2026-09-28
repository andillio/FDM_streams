"""GalaMilkyWayPotential in NumPy/CuPy: analytic acceleration, no JAX.

Matches streamsculptor.potential.GalaMilkyWayPotential (disk + bulge +
nucleus + halo) in kpc, Myr, Msun. Acceleration is -∇Φ for a test particle.
"""
import numpy as np
from astropy.constants import G as _G_const
from astropy import units as u

try:
	import cupy as cp
	CUPY_IMPORTED = True
except ImportError:
	cp = None
	CUPY_IMPORTED = False

# Same unit system as streamsculptor: kpc, Myr, Msun, radian
G_MW = float(_G_const.decompose([u.kpc, u.Myr, u.Msun]).value)

# GalaMilkyWayPotential parameters (streamsculptor/potential.py)
M_DISK = 6.80e10
A_DISK = 3.0
B_DISK = 0.28
M_BULGE = 5e9
C_BULGE = 1.0
M_NUCLEUS = 1.71e9
C_NUCLEUS = 0.07
M_HALO = 5.4e11
R_S_HALO = 15.62


def _xp(arr):
	if CUPY_IMPORTED and isinstance(arr, cp.ndarray):
		return cp
	return np


def _as_nx3(r, xp):
	r = xp.asarray(r)
	squeezed = False
	if r.ndim == 1:
		r = r.reshape(1, 3)
		squeezed = True
	return r, squeezed


def _miyamoto_nagai_potential(r, m, a, b, G, xp):
	R2 = r[:, 0]**2 + r[:, 1]**2
	z = r[:, 2]
	return -G * m / xp.sqrt(R2 + (xp.sqrt(z**2 + b**2) + a)**2)


def _miyamoto_nagai_acceleration(r, m, a, b, G, xp):
	x, y, z = r[:, 0], r[:, 1], r[:, 2]
	R2 = x**2 + y**2
	zeta = xp.sqrt(z**2 + b**2)
	D = a + zeta
	S = xp.sqrt(R2 + D**2)
	S3 = S**3
	pre = -G * m / S3
	acc = xp.empty_like(r)
	acc[:, 0] = pre * x
	acc[:, 1] = pre * y
	acc[:, 2] = pre * D * z / zeta
	return acc


def _hernquist_potential(r, m, r_s, G, xp):
	rad = xp.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2)
	return -G * m / (rad + r_s)


def _hernquist_acceleration(r, m, r_s, G, xp):
	rad = xp.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2)
	# a = -GM r_vec / (r (r + r_s)^2); r=0 → 0
	rad_safe = xp.maximum(rad, 1e-30)
	pre = -G * m / (rad_safe * (rad + r_s)**2)
	return pre[:, None] * r


def _nfw_potential(r, m, r_s, G, xp):
	rad = xp.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2)
	u = rad / r_s
	v_h2 = -G * m / r_s
	# log(1+u)/u → 1 as u → 0
	return xp.where(u < 1e-12, v_h2, v_h2 * xp.log1p(u) / u)


def _nfw_acceleration(r, m, r_s, G, xp):
	# Φ = -(GM/r) ln(1 + r/r_s)
	# a = GM [r/(r+r_s) - ln(1+r/r_s)] r_vec / r^3
	rad2 = r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2
	rad = xp.sqrt(rad2)
	rad_safe = xp.maximum(rad, 1e-30)
	u = rad / r_s
	bracket = rad / (rad + r_s) - xp.log1p(u)
	pre = G * m * bracket / (rad_safe**3)
	return pre[:, None] * r


class GalaMilkyWayPotential:
	"""Static MW potential; time argument is ignored."""

	def __init__(
		self,
		m_disk=M_DISK,
		a_disk=A_DISK,
		b_disk=B_DISK,
		m_bulge=M_BULGE,
		c_bulge=C_BULGE,
		m_nucleus=M_NUCLEUS,
		c_nucleus=C_NUCLEUS,
		m_halo=M_HALO,
		r_s_halo=R_S_HALO,
		G=G_MW,
	):
		self.m_disk = m_disk
		self.a_disk = a_disk
		self.b_disk = b_disk
		self.m_bulge = m_bulge
		self.c_bulge = c_bulge
		self.m_nucleus = m_nucleus
		self.c_nucleus = c_nucleus
		self.m_halo = m_halo
		self.r_s_halo = r_s_halo
		self.G = G

	def potential(self, r, t=None):
		xp = _xp(r)
		r, squeezed = _as_nx3(r, xp)
		G = self.G
		phi = (
			_miyamoto_nagai_potential(r, self.m_disk, self.a_disk, self.b_disk, G, xp)
			+ _hernquist_potential(r, self.m_bulge, self.c_bulge, G, xp)
			+ _hernquist_potential(r, self.m_nucleus, self.c_nucleus, G, xp)
			+ _nfw_potential(r, self.m_halo, self.r_s_halo, G, xp)
		)
		if squeezed:
			return phi[0]
		return phi

	def acceleration(self, r, t=None):
		"""Test-particle acceleration a = -∇Φ. Shape matches r: (3,) or (n, 3)."""
		xp = _xp(r)
		r, squeezed = _as_nx3(r, xp)
		G = self.G
		acc = (
			_miyamoto_nagai_acceleration(r, self.m_disk, self.a_disk, self.b_disk, G, xp)
			+ _hernquist_acceleration(r, self.m_bulge, self.c_bulge, G, xp)
			+ _hernquist_acceleration(r, self.m_nucleus, self.c_nucleus, G, xp)
			+ _nfw_acceleration(r, self.m_halo, self.r_s_halo, G, xp)
		)
		if squeezed:
			return acc[0]
		return acc


# Shared instance used by the CuPy solvers
gala_mw = GalaMilkyWayPotential()
