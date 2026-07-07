# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
"testRun_noFDM",
"testRun_yesFDM_m22=2",
# "testRun_yesFDM_m22=1",
"testRun_yesFDMHeavy",
 "testRun_forwards_m22=1",
# "testRun_yesFDM",
]
labels = [
# r'$f_\mathrm{FDM} = 10^{-6}$',
r'No FDM',
r'$m_\mathrm{22} = 2.0$',
r'$m_\mathrm{22} = 0.5$',
r'$m_\mathrm{22} = 1.0$',
# r'$m_\mathrm{22} = 0.2$',
]
final_drop = 10


def PlotStuff(d, fo):
	data_drops = d.data_drops

	r,v = d.LoadCorpData(data_drops)
	r_prog = np.load("../r_prog1.npy")
	rX = r[:,1] + r_prog[-1,1]
	rY = r[:,2] + r_prog[-1,2]

	fo.AddPlot(rX, rY, ls = ' ', mk = '.', alpha = 1, color = 'k')
	fo.SetTitle(labels[i])
	fo.SetXLabel(r'$x \, [\mathrm{kpc}]$')
	if i != 0:
		fo.RemoveYLabels()
	else:
		fo.SetYLabel(r'$y \, [\mathrm{kpc}]$')

	fo.SetXLim(-20,20)
	fo.SetYLim(-20,20)
	# fo.SetXLim(-16,16)
	# fo.SetYLim(-5,11)


if __name__ == "__main__":
	fo = pu.FigObj(len(simNames))
	for i in range(len(simNames)):
		name = simNames[i]
		d = do.MeshDataObj(name)
		PlotStuff(d, fo)
	fo.RemoveWhiteSpace()
	
	# fo.save('final_conditions_compare')
	fo.save('T=0.7Gyr_conditions_compare')
	fo.show()