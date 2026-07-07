# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
 "testRun_backwards_m22=1_run2",
 "testRun_forwards_m22=1_run2",
]
colors = ['r','b','g','c','m','y','k']


def GetSimulatedProg(d, backwards = False):
	t = np.linspace(0, d.Tf, d.data_drops + 1)

	if backwards:
		t = np.linspace(d.Tf, 0, d.data_drops + 1)

	r_prog = np.load(d.dataDir + "r_prog.npy")
	return t, r_prog

def GetImplicitProg(d):
	r_prog = np.load("../r_prog1.npy")
	t_prog = np.load("../t_prog1.npy")
	return t_prog, r_prog


def PlotTotalOrbit(d, fo, j):
	t_simulated, r_simulated = GetSimulatedProg(d, j == 0)

	if j == 0:
		t_implicit, r_implicit = GetImplicitProg(d)
		fo.AddPlot(r_implicit[:,2], r_implicit[:,1], color = 'k')
	fo.AddLine(r_simulated[:,2], r_simulated[:,1], color = colors[j], mk = 'o',
		ls = '')

def PlotOrbitPerturbation(d, fo, j):
	t_simulated, r_simulated = GetSimulatedProg(d, j == 0)
	t_implicit, r_implicit = GetImplicitProg(d)

	xVal = r_implicit[:,2]
	yVal = r_implicit[:,1]
	zVal = r_implicit[:,0]

	X_interped = np.interp(t_simulated, t_implicit, xVal)
	Y_interped = np.interp(t_simulated, t_implicit, yVal)
	Z_interped = np.interp(t_simulated, t_implicit, zVal)

	if j == 0:
		fo.AddPlot(r_simulated[:,2] - X_interped,
		 r_simulated[:,1] - Y_interped, color = colors[j])
	else:
		fo.AddLine(r_simulated[:,2] - X_interped,
		 r_simulated[:,1] - Y_interped, color = colors[j], ls = '', mk = 'o')

def PlotOrbitalRadius(d, fo, j):
	t_simulated, r_simulated = GetSimulatedProg(d, j == 0)
	R_simulated = np.sqrt(np.sum(np.abs(r_simulated)**2, axis = 1))

	if j == 0:
		t_implicit, r_implicit = GetImplicitProg(d)
		R_implicit = np.sqrt(np.sum(np.abs(r_implicit)**2, axis = 1))
		fo.AddPlot(t_implicit, R_implicit, color = 'k')		
	fo.AddLine(t_simulated, R_simulated, color = colors[j]
		, ls = '', mk = 'o')


if __name__ == "__main__":
	fo = pu.FigObj(3)

	# plot total orbit
	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotTotalOrbit(d, fo, i)
	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$y \, [\mathrm{kpc}]$')
	fo.SetTitle(r'Total Orbit')

	# plot orbit perturbation
	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotOrbitPerturbation(d, fo, i)
	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$y \, [\mathrm{kpc}]$')
	fo.SetTitle(r'Perturbed Orbit')

	# plot orbital radius
	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotOrbitalRadius(d, fo, i)
	fo.SetXLabel(r'$t \, [\mathrm{Myr}]$')
	fo.SetYLabel(r'$R \, [\mathrm{kpc}]$')
	fo.SetTitle(r'Orbital radius')

	fo.Save("plot_orbit_perturbations")
	fo.show()