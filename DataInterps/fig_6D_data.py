import numpy as np
import DataObj as do 
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric
import plotUtils as pu
import astroUtils as au

simName = "testRun_forwards_m22=1"
# simName = "testRun_yesFDMHeavy"


def ConvertToPhi(X, Y, Z):
	rX, rY, rZ = X * u.kpc, Y * u.kpc, Z * u.kpc
	galcen = SkyCoord(x=rX, y=rY, z=rZ, frame=Galactocentric)

	# Transform all at once
	icrs = galcen.transform_to('icrs')

	# Extract arrays of RA/Dec
	ra = icrs.ra.rad
	dec = icrs.dec.rad
	dist = icrs.distance.to(u.kpc).value

	Rot = np.zeros((3,3))
	Rot[0,:] = np.array([-0.47763, -0.17384, 0.86119])	
	Rot[1,:] = np.array([0.510845, -0.852445, 0.111245])	
	Rot[2,:] = np.array([0.714778, 0.493068, 0.49596])

	x_ = np.cos(ra)*np.cos(dec)
	y_ = np.sin(ra)*np.cos(dec)
	z_ = np.sin(dec)
	
	vec_ = np.zeros((len(rX), 3))
	vec_[:,0] = x_
	vec_[:,1] = y_
	vec_[:,2] = z_

	vec_rot = np.einsum("ij,nj->ni", Rot, vec_)
	x_rot = vec_rot[:,0]
	y_rot = vec_rot[:,1]
	z_rot = vec_rot[:,2]

	theta = np.pi/2. - np.arccos(z_rot)
	phi = np.arctan2(y_rot, x_rot)

	return phi, theta, dist


def GetCoords(d):
	r,v = d.LoadCorpData(d.data_drops)
	ts_back = np.load("../t_prog_orbit.npy")
	r_prog = np.load("../r_prog_orbit.npy")
	v_prog = np.load("../v_prog_orbit.npy")
	indexer = len(ts_back)//2
	# stream = np.load("../stream_final_conditions.npy")

	phi1, phi2, r = ConvertToPhi(r[:,0] + r_prog[indexer,0],
	 r[:,1] + r_prog[indexer,1],
	 r[:,2] + r_prog[indexer, 2])
	v_phi1, v_phi2, v_r = ConvertToPhi(v[:,0] + v_prog[indexer, 0],
	 v[:,1]+ v_prog[indexer, 1],
	 v[:,2]+ v_prog[indexer, 2])

	return phi1, phi2, r, v_phi1, v_phi2, v_r


def Plot6D(name):
	fo = pu.FigObj(2,3)
	rad2deg = 180/np.pi

	ts_back = np.load("../t_prog_orbit.npy")
	r_prog = np.load("../r_prog_orbit.npy")
	v_prog = np.load("../v_prog_orbit.npy")
	print(ts_back[len(ts_back)//2])

	indexer = len(ts_back)//2

	phi_prog, theta_prog, r_prog = ConvertToPhi(r_prog[:,0], r_prog[:,1], r_prog[:,2])
	v_phi_prog, v_theta_prog, v_r_prog = ConvertToPhi(v_prog[:,0], v_prog[:,1], v_prog[:,2])
	phi0 = np.array([phi_prog[indexer]])
	theta0 = np.array([theta_prog[indexer]])
	r0 = np.array([r_prog[indexer]])
	v_phi0 = np.array([v_phi_prog[indexer]])
	v_theta0 = np.array([v_theta_prog[indexer]])
	v_r0 = np.array([v_r_prog[indexer]])

	# radians per megayear to degrees per second
	rpmy2dps = rad2deg / (3.154e13)

	d = do.MeshDataObj(name)
	phi1, phi2, r, v_phi1, v_phi2, v_r = GetCoords(d)

	fo.AddPlot(phi1 * rad2deg, r - r0, ls = '', mk = 'o')
	fo.AddLine(phi0 * rad2deg, r0 * 0, color = 'r', ls = '', mk = 'o')
	fo.SetXLabel(r'$\phi_1$')
	fo.SetYLabel(r'$\Delta r \, [\mathrm{kpc}]$')

	fo.AddPlot(phi1 * rad2deg, (phi2 - theta0) * rad2deg, ls = '', mk = 'o')
	fo.AddLine(phi0 * rad2deg, theta0 * rad2deg, color = 'r', ls = '', mk = 'o')
	fo.SetXLabel(r'$\phi_1$')
	fo.SetYLabel(r'$\Delta \phi_2 \, [\mathrm{deg.}]$')

	fo.AddPlot(phi1 * rad2deg, (v_r - v_r0) / au.kms2kpcMyr, ls = '', mk = 'o')
	fo.AddLine(phi0 * rad2deg, v_r0 - v_r0, color = 'r', ls = '', mk = 'o')
	fo.SetXLabel(r'$\phi_1$')
	fo.SetYLabel(r'$\Delta v_r \, [\mathrm{km/s}]$')

	fo.AddPlot(phi1 * rad2deg, (v_phi1 - v_phi0) * rpmy2dps, ls = '', mk = 'o')
	fo.AddLine(phi0 * rad2deg, v_phi0 - v_phi0, color = 'r', ls = '', mk = 'o')
	fo.SetXLabel(r'$\phi_1$')
	fo.SetYLabel(r'$\Delta v_{\phi_1} \, [\mathrm{deg./s}]$')

	fo.AddPlot(phi1 * rad2deg, (v_phi2 - v_theta0) * rpmy2dps, ls = '', mk = 'o')
	fo.AddLine(phi0 * rad2deg, v_theta0 - v_theta0, color = 'r', ls = '', mk = 'o')
	fo.SetXLabel(r'$\phi_1$')
	fo.SetYLabel(r'$\Delta v_{\phi_2} \, [\mathrm{deg./s}] $')

	fo.AddHist(phi1 * rad2deg, nBins= 100, density = True)
	fo.SetXLabel(r'$\phi_1$')
	fo.SetYLabel(r'$\rho \, [\mathrm{stars / deg.}] $')
	fo.save(d.dataDir + "fig_6D")

	fo = pu.FigObj()

	n, edges = np.histogram(phi1 * rad2deg, bins = 100, density = True)
	x_ = .5*(edges[0:len(edges)-1] + edges[1:])

	fft = np.fft.fft(n)
	x_range = np.max(x_) - np.min(x_)
	kx = np.fft.fftfreq(len(n), d = x_range / len(x_))
	rho_k = np.abs(fft)**2

	fo.AddPlot(kx[kx > 0], rho_k[kx > 0])
	fo.SetXLabel(r'$kx$')
	fo.SetYLabel(r'$\rho_k$')
	fo.SetLogLog(kx[kx > 0], rho_k[kx > 0])

	fo.show()



if __name__ == "__main__":
	Plot6D(simName)