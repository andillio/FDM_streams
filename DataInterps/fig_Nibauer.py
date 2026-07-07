# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simName = "testRun_noFDM"
# simName = "testRun_yesFDM"
# simName = "testRun_yesFDMHeavy"
# simName = "testRun_yesFDM_m22=1"
# simName = "testRun_yesFDM_m22=2"


def PlotAngle(fo, d, j, bounds):
	phi1 = np.load(d.dataDir + "phi1_deg.npy")
	phi2 = np.load(d.dataDir + "phi2_deg.npy")

	phi2 = phi2[(phi1 > bounds[0]) & (phi1 < bounds[1])]
	# phi2 -= np.mean(phi2)
	phi1 = phi1[(phi1 > bounds[0]) & (phi1 < bounds[1])]

	fo.AddPlot(phi1, phi2, ls= '', mk ='.', color = 'k')
	fo.SetYLim(-2,2)

	if j != 0:
		fo.RemoveYLabels()

def PlotRadialVel(fo, d, j, bounds):
	phi1 = np.load(d.dataDir + "phi1_deg.npy")
	v_r = np.load(d.dataDir + "v_r.npy")

	v_r_cut = v_r[(phi1 > bounds[0]) & (phi1 < bounds[1])] # select region
	v_r_cut -= np.mean(v_r_cut)
	phi1 = phi1[(phi1 > bounds[0]) & (phi1 < bounds[1])]

	x_eval = np.linspace(bounds[0], bounds[1], 100)
	y_fit = mu.polyfit(x_eval, phi1, v_r_cut, 2)

	fo.AddPlot(phi1, v_r_cut, ls= '', mk ='.', color = 'k')
	fo.AddLine(x_eval, y_fit,color = 'r')
	fo.SetYLim(-50,50)
	fo.SetXLim(bounds[0],bounds[1])

	if j != 0:
		fo.RemoveYLabels()


def PlotSigma(fo, d, j, bounds, cut = None):
	phi1 = np.load(d.dataDir + "phi1_deg.npy")
	v_r = np.load(d.dataDir + "v_r.npy")

	v_r_cut = v_r[(phi1 > bounds[0]) & (phi1 < bounds[1])] # select region
	v_r_cut -= np.mean(v_r_cut)
	phi1 = phi1[(phi1 > bounds[0]) & (phi1 < bounds[1])]

	if not(cut is None):
		v_r_cut = v_r_cut[v_r_cut < cut]

	phi_mid = np.mean(bounds)
	sig = np.std(v_r_cut)
	N = len(v_r_cut)
	se_s = sig / np.sqrt(2 * (N - 1))

	fo.AddPlot([phi_mid], [sig], ls= '', mk ='o', color = 'k')
	fo.AddLine([phi_mid, phi_mid], [sig - se_s, sig + se_s], color = 'k')
	fo.SetYLim(-50,50)
	fo.SetXLim(bounds[0],bounds[1])
	fo.SetXLabel(r'$\phi_1 \, [\mathrm{deg}]$')


if __name__ == "__main__":
	fo = pu.FigObj(3,3)

	d = do.MeshDataObj(simName)

	PlotAngle(fo, d,0, [-60,-40])
	fo.SetYLabel(r'$\phi_2 \, [\mathrm{deg}]$')
	PlotAngle(fo, d,1, [-40,-20])
	PlotAngle(fo, d,2, [-20,-0])

	PlotRadialVel(fo, d,0, [-60,-40])
	fo.SetYLabel(r'$v_r \, [\mathrm{km/s}]$')
	PlotRadialVel(fo, d,1, [-40,-20])
	PlotRadialVel(fo, d,2, [-20,-0])

	PlotSigma(fo, d,0, [-60,-40])
	fo.SetYLabel(r'$\sigma_{v_r} \, [\mathrm{km/s}]$')
	PlotSigma(fo, d,1, [-40,-20])
	PlotSigma(fo, d,2, [-20,-0])

	fo.RemoveWhiteSpace()
	fo.Save('Nibauer_with_fit_line')
	fo.show()
