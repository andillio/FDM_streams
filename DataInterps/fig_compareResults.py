# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
"testRun_noFDM",
"testRun_yesFDMHeavy",
"testRun_yesFDM",
]
labels = [
# r'$f_\mathrm{FDM} = 10^{-6}$',
r'No FDM (Andrews sim)',
r'$m_\mathrm{22} = 0.5$',
r'$m_\mathrm{22} = 0.2$',
]
final_drop = -1


def PlotReference(fo):
	stream = np.load("stream_final_conditions.npy")
	rX, rY = stream[:,1], stream[:,2]
	fo.AddPlot(rX, rY, ls = ' ', mk = '.', alpha = 1, color = 'k')
	fo.SetTitle(r'No FDM (Keirs sim)')
	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$y \, [\mathrm{kpc}]$')

	fo.SetXLim(-16,16)
	fo.SetYLim(-5,11)
	fo.SetXLim(-16,16)
	fo.SetYLim(-16,16)


def PlotStuff(d, fo):
	data_drops = d.data_drops

	r,v = d.LoadCorpData(data_drops)
	r_prog = np.load("../r_prog1.npy")
	rX = r[:,1] + r_prog[-1,1]
	rY = r[:,2] + r_prog[-1,2]

	fo.AddPlot(rX, rY, ls = ' ', mk = '.', alpha = 1, color = 'k')
	fo.SetTitle(labels[i])
	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	fo.SetYLabel(r'$y \, [\mathrm{kpc}]$')

	fo.SetXLim(-16,16)
	fo.SetYLim(-16,16)
	# fo.SetXLim(-16,16)
	# fo.SetYLim(-5,11)


if __name__ == "__main__":
	fo = pu.FigObj(len(simNames) + 1)
	PlotReference(fo)
	for i in range(len(simNames)):
		name = simNames[i]
		d = do.MeshDataObj(name)
		PlotStuff(d, fo)
	
	fo.save('final_conditions_compare')
	fo.show()