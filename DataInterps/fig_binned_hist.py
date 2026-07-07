# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
 "testRun_noFDM",
 "testRun_yesFDM_m22=1",
 "testRun_yesFDM_m22=2",
]
colors = ['r','b','g','c','m','y','k']


def PlotAngle(d, j, bounds, cut):
	phi1 = np.load(d.dataDir + "phi1_deg.npy")
	v_r = np.load(d.dataDir + "v_r.npy")

	v_r_cut = v_r[(phi1 > bounds[0]) & (phi1 < bounds[1])] # select region
	v_r_cut -= np.mean(v_r_cut)
	v_r_cut = v_r_cut[np.abs(v_r_cut) < cut]
	# print(v_r)
	print(np.std(v_r_cut))


if __name__ == "__main__":
	fo = pu.FigObj(3,2)

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotAngle(d, i, [-60,-40], 30)

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotAngle(d, i, [-40,-20], 30)

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotAngle(d, i, [-20,0], 30)

	fo.Save("plot_orbit")
	fo.show()