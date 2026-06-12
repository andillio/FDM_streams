# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
 "testRun_backwards_m22=1",
 "testRun_forwards_m22=1",
]
colors = ['r','b','g','c','m','y','k']


def TotalOrbit(d, fo, j):
	N = d.N
	T = d.Tf
	L = d.L 
	dx = L/N

	r_orbit = np.zeros((d.data_drops+1, 3))
	t = np.linspace(0, T, d.data_drops + 1)

	r_prog = np.load("../r_prog1.npy")
	if j == 0:
		r_prog = np.flip(np.load("../r_prog1.npy"), axis = 0)
		for i in range(d.data_drops+1):
			r,v = d.LoadCorpData(i, center = False)
			r_orbit[i,:] = r[0] 
			t[i] = i * T / d.data_drops
	else:
		r_orbit = np.load(d.dataDir + "r_perturb.npy")
		# print(r_orbit)
	t_prog = np.load("../t_prog1.npy")
	r_prog0 = np.interp(t, t_prog, r_prog[:,1])
	r_prog2 = np.interp(t, t_prog, r_prog[:,2])
	print(r_prog[0])

	if j == 0:
		fo.AddLine(r_orbit[:,2] + r_prog2, r_orbit[:,1] + r_prog0, color = colors[j])
	else:
		fo.AddLine(r_orbit[:,2] + r_prog2, r_orbit[:,1] + r_prog0,
		 color = colors[j], ls = '', mk = 'o')


def OrbitPerturbation(d, fo, j):
	N = d.N
	T = d.Tf
	L = d.L 
	dx = L/N

	r_orbit = np.zeros((d.data_drops+1, 3))
	t = np.zeros(d.data_drops+1)

	if j == 0:
		for i in range(d.data_drops+1):
			r,v = d.LoadCorpData(i, center = False)
			r_orbit[i,:] = r[0] 
			t[i] = i * T / d.data_drops
	else:
		r_orbit = np.load(d.dataDir + "r_perturb.npy")

	if j == 0:
		fo.AddPlot(r_orbit[:,2], r_orbit[:,1], color = colors[j])
	else:
		fo.AddLine(r_orbit[:,2], r_orbit[:,1], color = colors[j], ls = '', mk = 'o')



def OrbitalRadius(d, fo, j):
	N = d.N
	T = d.Tf
	L = d.L 
	dx = L/N

	r_orbit = np.zeros((d.data_drops+1, 3))
	t = np.linspace(0, T, d.data_drops + 1)

	r_prog = np.load("../r_prog1.npy")
	if j == 0:
		r_prog = np.flip(np.load("../r_prog1.npy"), axis = 0)
		for i in range(d.data_drops+1):
			r,v = d.LoadCorpData(i, center = False)
			r_orbit[i,:] = r[0] 
			t[i] = i * T / d.data_drops
	else:
		r_orbit = np.load(d.dataDir + "r_perturb.npy")
	t_prog = np.load("../t_prog1.npy")

	r_prog_interped = np.zeros((len(t), 3))
	r_prog_interped[:,0] = np.interp(t, t_prog, r_prog[:,0])
	r_prog_interped[:,1] = np.interp(t, t_prog, r_prog[:,1])
	r_prog_interped[:,2] = np.interp(t, t_prog, r_prog[:,2])

	r_total = r_orbit + r_prog_interped
	R = np.sqrt(np.sum(np.abs(r_total)**2, axis = 1))
	R_prog = np.sqrt(np.sum(np.abs(r_prog_interped)**2, axis = 1))

	if j == 0:
		fo.AddPlot(t, R, color = colors[j])
		fo.AddLine(t, R_prog, color = 'k')
	else:
		fo.AddLine(t, R, color = colors[j], ls = '', mk = 'o')


if __name__ == "__main__":
	fo = pu.FigObj(3)
	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		TotalOrbit(d, fo, i)
	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$y \, [\mathrm{kpc}]$')
	fo.SetTitle(r'Total Orbit')

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		OrbitPerturbation(d, fo, i)
	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$y \, [\mathrm{kpc}]$')
	fo.SetTitle(r'Perturbed Orbit')

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		OrbitalRadius(d, fo, i)
	fo.SetXLabel(r'$t \, [\mathrm{Myr}]$')
	fo.SetYLabel(r'$R \, [\mathrm{kpc}]$')
	fo.SetTitle(r'Orbital radius')

	fo.Save("plot_orbit")
	fo.show()