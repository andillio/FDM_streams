# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
"testRun_noFDM",
# "testRun_yesFDM_m22=1",
"testRun_yesFDMHeavy",
"testRun_yesFDM",
]
labels = [
# r'$f_\mathrm{FDM} = 10^{-6}$',
r'No FDM',
# r'$m_\mathrm{22} = 1.0$',
r'$m_\mathrm{22} = 0.5$',
r'$m_\mathrm{22} = 0.2$',
]
colors = ['r','b', 'g','c','m','y']



def PlotStuff(d, fo, j):
	data_drops = d.data_drops

	r,v = d.LoadCorpData(data_drops)
	r_prog = np.load("../r_prog1.npy")
	rX = r[:,1] + r_prog[-1,1]
	rY = r[:,2] + r_prog[-1,2]

	r_prof_final = r_prog[-1]

	bin_edges = np.linspace(0,np.pi,100)

	angles = []
	for i in range(len(r)):
		r_ = r[i]
		angle_ = mu.AngleBetweenVectors(r_, r_prof_final)
		angles.append(angle_)
	
	n, edges = np.histogram(angles, bins = bin_edges)
	x_ = .5*(edges[0:len(edges)-1] + edges[1:])

	if j == 0:
		fo.AddPlot(x_, n, label = labels[j], color = colors[j])
	else:
		fo.AddLine(x_, n, label = labels[j], color = colors[j])



if __name__ == "__main__":
	fo = pu.FigObj()
	for i in range(len(simNames)):
		name = simNames[i]
		d = do.MeshDataObj(name)
		PlotStuff(d, fo, i)
	fo.legend()
	fo.RemoveWhiteSpace()
	
	fo.save('angle_distribution_compare')
	fo.show()