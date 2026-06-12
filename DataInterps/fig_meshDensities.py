# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import matplotlib.pyplot as plt 
import numpy as np 

# simName = "ics_2_5m22"
# simName = "solitonData_light"
simName = "testRun_yesFDM"
simName = "testRun_backwards_m22=1"

def PlotStuff(d):
	initial_drop = 0
	mid_drop = d.data_drops // 2
	final_drop = d.data_drops 

	data_drops = d.data_drops

	N = d.N
	T = d.Tf
	# print(d.params)
	# print(d.Tf, d.N)
	L = d.L 
	dx = L/N
	x = [-L/2., L/2.]

	fo = pu.FigObj(3,2)

	psi = d.LoadPsi(initial_drop)
	print(np.sum(np.abs(psi)**2)*dx**3)
	slice_ = d.GetFieldDensity()[N//2+1,:,:]
	# fo.AddDens2d(x, np.log10(slice_) )
	fo.AddDens2d(x, slice_ )
	T_i = T * float(initial_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_i))
	fo.RemoveXLabels()

	psi = d.LoadPsi(mid_drop)
	slice_ = d.GetFieldDensity()[N//2+1,:,:]
	print(np.sum(np.abs(psi)**2)*dx**3)
	print(np.sum(np.abs(psi)**2)*dx**3 / L**3)
	# fo.AddDens2d(x, np.log10(slice_) )
	fo.AddDens2d(x, slice_ )
	T_m = T * float(mid_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_m))
	fo.RemoveXLabels()
	fo.RemoveYLabels()	
	fo.SetTitle(r"ULDM halo density")


	psi = d.LoadPsi(final_drop)
	slice_ = d.GetFieldDensity()[N//2+1,:,:]
	print(np.sum(np.abs(psi)**2)*dx**3 / L**3)
	# ax, im1 = fo.AddDens2d(x, np.log10(slice_) )
	fo.AddDens2d(x, slice_ )
	print(np.sum(np.abs(psi)**2)*dx**3 / L**3)
	T_f = T * float(final_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_f))
	fo.RemoveXLabels()
	fo.RemoveYLabels()


	# fo.AddTextRightLabel(r'Density slices')

	psi = d.LoadPsi(initial_drop)
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	fo.AddDens2d(x, np.log10(proj))
	# ax, im2 = fo.AddDens2d(x, proj)

	psi = d.LoadPsi(mid_drop)
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	fo.AddDens2d(x, np.log10(proj))
	# ax, im2 = fo.AddDens2d(x, proj)
	fo.RemoveYLabels()

	psi = d.LoadPsi(final_drop)
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	ax, im2 = fo.AddDens2d(x, np.log10(proj) )
	fo.RemoveYLabels()
	# fo.AddColorbar(im2)
	# fo.AddTextRightLabel(r'Density projections')

	fo.RemoveWhiteSpace()

	fo.save(d.dataDir + 'densities')


def Main(name):
	d = do.MeshDataObj(name)
	PlotStuff(d)


if __name__ == "__main__":
	Main(simName)
	plt.show()