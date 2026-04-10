# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as sp2

simName = "testRun2"

def PlotStuff(d):
	fo = pu.FigObj(2)

	v2 = d.get_K(d.N, d.L)*d.hbar_[0]**2
	v_bar = np.zeros(d.data_drops + 1)
	t = np.linspace(0,d.Tf, d.data_drops + 1)

	for i in range(d.data_drops + 1):
		psi = d.LoadPsi(i)
		rho_k = np.abs(d.GetFFt(psi))**2
		rho_k /= np.sum(rho_k)
		mu = np.sum(np.sqrt(v2)*rho_k)
		a = mu / (np.sqrt(8/np.pi))
		v_bar[i] = np.sqrt(2)*a

	v_bar_kms = v_bar / au.kms2kpcMyr
	print(v_bar_kms)
	print(v_bar_kms[-1])
	fo.AddPlot(t, v_bar_kms)
	fo.SetXLabel(r'$t \, [\mathrm{Myr}]$')
	fo.SetYLabel(r'$|v_s| \, [\mathrm{km/s}]$')
	fo.SetYLim(0, np.max(v_bar_kms)*1.1)
	fo.SetXLim(0, d.Tf)

	psi = d.LoadPsi(2)
	rho_k = np.abs(d.GetFFt(psi))**2
	rho_k /= np.sum(rho_k)
	

	v_flat = (np.sqrt(v2)).reshape(-1) / au.kms2kpcMyr
	rho_flat = rho_k.reshape(-1)

	fo.AddPlot(v_flat, rho_flat, ls = '', mk = 'o')
	# fo.SetYLim(0, 1e-5)
	fo.show()



def Main(name):
	d = do.MeshDataObj(name)
	PlotStuff(d)


if __name__ == "__main__":
	Main(simName)
	plt.show()