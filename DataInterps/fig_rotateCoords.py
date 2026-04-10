from cupy import rad2deg
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric
import plotUtils as pu

t_buffer = 100 # Myr


def ConvertToPhi(X, Y, Z):
	rX, rY, rZ = X * u.kpc, Y * u.kpc, Z * u.kpc
	galcen = SkyCoord(x=rX, y=rY, z=rZ, frame=Galactocentric)

	# Transform all at once
	icrs = galcen.transform_to('icrs')

	# Extract arrays of RA/Dec
	ra = icrs.ra.rad
	dec = icrs.dec.rad

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

	return phi, theta


def PlotStars(fo):
	stream = np.load("stream_final_conditions.npy")
	rX, rY, rZ = stream[:,0] * u.kpc, stream[:,1] * u.kpc, stream[:,2] * u.kpc

	phi, theta = ConvertToPhi(stream[:,0], stream[:,1], stream[:,2])
	rad2deg = 180/np.pi

	# fo.AddPlot(ra,dec, ls = '', mk = 'o')
	fo.AddPlot(phi * rad2deg, theta * rad2deg, ls = '', mk = 'o')
	fo.SetXLabel(r'$\phi_1$')
	fo.SetYLabel(r'$\phi_2$')


def PlotOrbit(fo):
	ts_back = np.load("../t_prog_orbit.npy")
	r_prog = np.load("../r_prog_orbit.npy")
	print(ts_back[len(ts_back)//2])

	indexer = len(ts_back)//2

	# r_prog = r_prog[len(r_prog) - 100:]
	# r_prog = r_prog[ts_back > np.max(ts_back)-t_buffer]
	# ts_back = ts_back[ts_back > np.max(ts_back)-t_buffer]


	phi, theta = ConvertToPhi(r_prog[:,0], r_prog[:,1], r_prog[:,2])
	phi0 = np.array([phi[indexer]])
	theta0 = np.array([theta[indexer]])

	rad2deg = 180/np.pi

	# fo.AddPlot(ra,dec, ls = '', mk = 'o')
	fo.AddLine(phi * rad2deg, theta * rad2deg, color = 'k')
	fo.AddLine(phi0 * rad2deg, theta0 * rad2deg, color = 'r', ls = '', mk = 'o')


if __name__ == "__main__":
	fo = pu.FigObj()
	PlotStars(fo)
	PlotOrbit(fo)
	fo.Save('rotated_coords')
	fo.show()