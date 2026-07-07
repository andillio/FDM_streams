# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
 "testRun_yesFDM",
 "testRun_yesFDMHeavy",
 # "testRun_yesFDM_m22=1",
 "testRun_yesFDM_m22=2",
 "testRun_forwards_m22=1",
]
colors = ['r','b','g','c','m','y','k']
mass_range = [1.3e-1, 3]


def PlotSigma(fo, d, j, bounds, cut = None):
	phi1 = np.load(d.dataDir + "phi1_deg.npy")
	v_r = np.load(d.dataDir + "v_r.npy")

	v_r_cut = v_r[(phi1 > bounds[0]) & (phi1 < bounds[1])] # select region
	v_r_cut -= np.mean(v_r_cut)
	phi1 = phi1[(phi1 > bounds[0]) & (phi1 < bounds[1])]
	v_r_cut = v_r_cut - mu.polyfit(phi1, phi1, v_r_cut, 2)

	if not(cut is None):
		v_r_cut = v_r_cut[v_r_cut < cut]

	sig = np.std(v_r_cut)
	N = len(v_r_cut)
	se_s = sig / np.sqrt(2 * (N - 1))
	print(sig)

	if j == 0:
		fo.AddPlot([d.m22], [sig], ls= '', mk ='o', color = 'k', label = r'simulated data')
	else:
		fo.AddLine([d.m22], [sig], ls= '', mk ='o', color = 'k')
	fo.AddLine([d.m22, d.m22], [sig - se_s, sig + se_s], color = 'k')


def PlotUncertainty(fo, lower, upper):
	ax, im = fo.AddHorLine(lower, ls = '-', color = 'r', label = r'$\sigma_{vr}$')
	ax, im = fo.AddHorLine(upper, ls = '-', color = 'r')
	ax.axhspan(lower, upper, color="r", alpha=0.3)
	# ax.axhspan(0, beta, color="r", alpha=0.3)



if __name__ == "__main__":
	fo = pu.FigObj(3)

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotSigma(fo, d,i, [-60,-40], 30)

	fo.SetXLabel(r'$m_{22}$')
	fo.SetYLabel(r'$\sigma_{vr}$')
	fo.SetXLog(mass_range)
	fo.SetYLim(0, 26)
	PlotUncertainty(fo, 0, 4.2)
	fo.legend()
	fo.SetTitle(r'$\phi_1 \in [-60,-40]$')

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotSigma(fo, d,i, [-40,-20], 30)

	fo.SetXLabel(r'$m_{22}$')
	fo.SetXLog(mass_range)
	fo.SetYLim(0, 26)
	PlotUncertainty(fo, 2, 6.5)
	fo.SetTitle(r'$\phi_1 \in [-40,-20]$')
	fo.RemoveYLabels()

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotSigma(fo, d,i, [-20, 0], 30)

	fo.SetXLabel(r'$m_{22}$')
	fo.SetXLog(mass_range)
	fo.SetYLim(0, 26)
	PlotUncertainty(fo, 0, 6.1)
	fo.SetTitle(r'$\phi_1 \in [-20, 0]$')
	fo.RemoveYLabels()

	fo.RemoveWhiteSpace()
	fo.Save("plot_sigma_vs_mass_30")
	fo.show()