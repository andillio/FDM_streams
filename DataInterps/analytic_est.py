# pylint: disable=C,W
import astroUtils as au
from astropy.units import M_e
import numpy as np 

rho_dm = 1e7
sigma_dm = 200 * au.kms2kpcMyr
hbar = au.hbar
t = 3.5e3
var_v = 3 * au.kms2kpcMyr 

m3 = np.pi**3 * au.G**2 * hbar**3 * rho_dm**2 * t / sigma_dm**4 / var_v
m = (m3)**(1/3.)
m /= au.eV2SolarMass
print(m)

# lam = np.pi * hbar / sigma_dm / (m*au.eV2SolarMass * 10)
# M_eff = rho_dm * lam**3 
# print(M_eff)
# print(np.log10(M_eff))
