# Titan Robotics Team 2022: CPU fitting models
# Written by Dev Singh
# Notes:
#   this module is cuda-optimized (as appropriate) and vectorized (except for one small part)
# setup:

__version__ = "0.0.2"

# changelog should be viewed using print(analysis.fits.__changelog__)
__changelog__ = """changelog:
	0.0.2:
		- renamed module to Fit
	0.0.1:
		- initial release, add circle fitting with LSC
"""

__author__ = (
	"Dev Singh <dev@devksingh.com>"
)

__all__ = [
	'CircleFit'
]

import numpy as np

class CircleFit:
	"""Class to fit data to a circle using the Least Square Circle (LSC) method"""
	# For more information on the LSC method, see: 
	# http://www.dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf
	def __init__(self, x, y, xy=None):
		self.ournp = np #todo: implement cupy correctly
		if type(x) == list:
			x = np.array(x)
		if type(y) == list:
			y = np.array(y)
		if type(xy) == list:
			xy = np.array(xy)
		if xy != None: 
			self.coords = xy
		else: 
			# following block combines x and y into one array if not already done
			self.coords = self.ournp.vstack(([x.T], [y.T])).T
	def calc_R(x, y, xc, yc):
		"""Returns distance between center and point"""
		return self.ournp.sqrt((x-xc)**2 + (y-yc)**2)
	def f(c, x, y):
		"""Returns distance between point and circle at c"""
		Ri = calc_R(x, y, *c)
		return Ri - Ri.mean()
	def LSC(self):
		"""Fits given data to a circle and returns the center, radius, and variance"""
		x = self.coords[:, 0]
		y = self.coords[:, 1]
		# guessing at a center
		x_m = self.ournp.mean(x)
		y_m = self.ournp.mean(y)

		# calculation of the reduced coordinates
		u = x - x_m
		v = y - y_m

		# linear system defining the center (uc, vc) in reduced coordinates:
		#    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
		#    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
		Suv  = self.ournp.sum(u*v)
		Suu  = self.ournp.sum(u**2)
		Svv  = self.ournp.sum(v**2)
		Suuv = self.ournp.sum(u**2 * v)
		Suvv = self.ournp.sum(u * v**2)
		Suuu = self.ournp.sum(u**3)
		Svvv = self.ournp.sum(v**3)

		# Solving the linear system
		A = self.ournp.array([ [ Suu, Suv ], [Suv, Svv]])
		B = self.ournp.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
		uc, vc = self.ournp.linalg.solve(A, B)

		xc_1 = x_m + uc
		yc_1 = y_m + vc

		# Calculate the distances from center (xc_1, yc_1)
		Ri_1     = self.ournp.sqrt((x-xc_1)**2 + (y-yc_1)**2)
		R_1      = self.ournp.mean(Ri_1)
		# calculate residual error
		residu_1 = self.ournp.sum((Ri_1-R_1)**2)
		return (xc_1, yc_1, R_1, residu_1)