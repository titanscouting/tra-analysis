# Titan Robotics Team 2022: CUDA-based Regressions Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#   this module has been automatically inegrated into analysis.py, and should be callable as a class from the package
#   this module is cuda-optimized (as appropriate) and vectorized (except for one small part)
# setup:

__version__ = "0.0.5"

# changelog should be viewed using print(analysis.regression.__changelog__)
__changelog__ = """
	0.0.5:
		- add circle fitting with LSC and HyperFit
	0.0.4:
		- bug fixes
		- fixed changelog
	0.0.3:
		- bug fixes
	0.0.2:
		-Added more parameters to log, exponential, polynomial
		-Added SigmoidalRegKernelArthur, because Arthur apparently needs
		to train the scaling and shifting of sigmoids
	0.0.1:
		-initial release, with linear, log, exponential, polynomial, and sigmoid kernels
		-already vectorized (except for polynomial generation) and CUDA-optimized
"""

__author__ = (
	"Jacob Levine <jlevine@imsa.edu>",
	"Arthur Lu <learthurgo@gmail.com>",
	"Dev Singh <dev@devksingh.com>"
)

__all__ = [
	'factorial',
	'take_all_pwrs',
	'num_poly_terms',
	'set_device',
	'LinearRegKernel',
	'SigmoidalRegKernel',
	'LogRegKernel',
	'PolyRegKernel',
	'ExpRegKernel',
	'SigmoidalRegKernelArthur',
	'SGDTrain',
	'CustomTrain',
	'CircleFit'
]

import torch
import numpy as np


global device

device = "cuda:0" if torch.torch.cuda.is_available() else "cpu"

if device !== "cpu":
	import cupy as cp

#todo: document completely

def set_device(self, new_device):
	device=new_device

class LinearRegKernel():
	parameters= []
	weights=None
	bias=None
	def __init__(self, num_vars):
		self.weights=torch.rand(num_vars, requires_grad=True, device=device)
		self.bias=torch.rand(1, requires_grad=True, device=device)
		self.parameters=[self.weights,self.bias]
	def forward(self,mtx):
		long_bias=self.bias.repeat([1,mtx.size()[1]])
		return torch.matmul(self.weights,mtx)+long_bias

class SigmoidalRegKernel():
	parameters= []
	weights=None
	bias=None
	sigmoid=torch.nn.Sigmoid()
	def __init__(self, num_vars):
		self.weights=torch.rand(num_vars, requires_grad=True, device=device)
		self.bias=torch.rand(1, requires_grad=True, device=device)
		self.parameters=[self.weights,self.bias]
	def forward(self,mtx):
		long_bias=self.bias.repeat([1,mtx.size()[1]])
		return self.sigmoid(torch.matmul(self.weights,mtx)+long_bias)

class SigmoidalRegKernelArthur():
	parameters= []
	weights=None
	in_bias=None
	scal_mult=None
	out_bias=None
	sigmoid=torch.nn.Sigmoid()
	def __init__(self, num_vars):
		self.weights=torch.rand(num_vars, requires_grad=True, device=device)
		self.in_bias=torch.rand(1, requires_grad=True, device=device)
		self.scal_mult=torch.rand(1, requires_grad=True, device=device)
		self.out_bias=torch.rand(1, requires_grad=True, device=device)
		self.parameters=[self.weights,self.in_bias, self.scal_mult, self.out_bias]
	def forward(self,mtx):
		long_in_bias=self.in_bias.repeat([1,mtx.size()[1]])
		long_out_bias=self.out_bias.repeat([1,mtx.size()[1]])
		return (self.scal_mult*self.sigmoid(torch.matmul(self.weights,mtx)+long_in_bias))+long_out_bias

class LogRegKernel():
	parameters= []
	weights=None
	in_bias=None
	scal_mult=None
	out_bias=None
	def __init__(self, num_vars):
		self.weights=torch.rand(num_vars, requires_grad=True, device=device)
		self.in_bias=torch.rand(1, requires_grad=True, device=device)
		self.scal_mult=torch.rand(1, requires_grad=True, device=device)
		self.out_bias=torch.rand(1, requires_grad=True, device=device)
		self.parameters=[self.weights,self.in_bias, self.scal_mult, self.out_bias]
	def forward(self,mtx):
		long_in_bias=self.in_bias.repeat([1,mtx.size()[1]])
		long_out_bias=self.out_bias.repeat([1,mtx.size()[1]])
		return (self.scal_mult*torch.log(torch.matmul(self.weights,mtx)+long_in_bias))+long_out_bias

class ExpRegKernel():
	parameters= []
	weights=None
	in_bias=None
	scal_mult=None
	out_bias=None
	def __init__(self, num_vars):
		self.weights=torch.rand(num_vars, requires_grad=True, device=device)
		self.in_bias=torch.rand(1, requires_grad=True, device=device)
		self.scal_mult=torch.rand(1, requires_grad=True, device=device)
		self.out_bias=torch.rand(1, requires_grad=True, device=device)
		self.parameters=[self.weights,self.in_bias, self.scal_mult, self.out_bias]
	def forward(self,mtx):
		long_in_bias=self.in_bias.repeat([1,mtx.size()[1]])
		long_out_bias=self.out_bias.repeat([1,mtx.size()[1]])
		return (self.scal_mult*torch.exp(torch.matmul(self.weights,mtx)+long_in_bias))+long_out_bias

class PolyRegKernel():
	parameters= []
	weights=None
	bias=None
	power=None
	def __init__(self, num_vars, power):
		self.power=power
		num_terms=self.num_poly_terms(num_vars, power)
		self.weights=torch.rand(num_terms, requires_grad=True, device=device)
		self.bias=torch.rand(1, requires_grad=True, device=device)
		self.parameters=[self.weights,self.bias]
	def num_poly_terms(self,num_vars, power):
		if power == 0:
			return 0
		return int(self.factorial(num_vars+power-1) / self.factorial(power) / self.factorial(num_vars-1)) + self.num_poly_terms(num_vars, power-1)
	def factorial(self,n):
		if n==0:
			return 1
		else:
			return n*self.factorial(n-1)
	def take_all_pwrs(self, vec, pwr):
		#todo: vectorize (kinda)
		combins=torch.combinations(vec, r=pwr, with_replacement=True)
		out=torch.ones(combins.size()[0]).to(device).to(torch.float)
		for i in torch.t(combins).to(device).to(torch.float):
			out *= i
		if pwr == 1:
			return out
		else:
			return torch.cat((out,self.take_all_pwrs(vec, pwr-1)))
	def forward(self,mtx):
		#TODO: Vectorize the last part
		cols=[]
		for i in torch.t(mtx):
			cols.append(self.take_all_pwrs(i,self.power))
		new_mtx=torch.t(torch.stack(cols))
		long_bias=self.bias.repeat([1,mtx.size()[1]])
		return torch.matmul(self.weights,new_mtx)+long_bias

def SGDTrain(self, kernel, data, ground, loss=torch.nn.MSELoss(), iterations=1000, learning_rate=.1, return_losses=False):
	optim=torch.optim.SGD(kernel.parameters, lr=learning_rate)
	data_cuda=data.to(device)
	ground_cuda=ground.to(device)
	if (return_losses):
		losses=[]
		for i in range(iterations):
			with torch.set_grad_enabled(True):
				optim.zero_grad()
				pred=kernel.forward(data_cuda)
				ls=loss(pred,ground_cuda)
				losses.append(ls.item())
				ls.backward()
				optim.step()
		return [kernel,losses]
	else:
		for i in range(iterations):
			with torch.set_grad_enabled(True):
				optim.zero_grad()
				pred=kernel.forward(data_cuda)
				ls=loss(pred,ground_cuda)
				ls.backward()
				optim.step()
		return kernel

def CustomTrain(self, kernel, optim, data, ground, loss=torch.nn.MSELoss(), iterations=1000, return_losses=False):
	data_cuda=data.to(device)
	ground_cuda=ground.to(device)
	if (return_losses):
		losses=[]
		for i in range(iterations):
			with torch.set_grad_enabled(True):
				optim.zero_grad()
				pred=kernel.forward(data)
				ls=loss(pred,ground)
				losses.append(ls.item())
				ls.backward()
				optim.step()
		return [kernel,losses]
	else:
		for i in range(iterations):
			with torch.set_grad_enabled(True):
				optim.zero_grad()
				pred=kernel.forward(data_cuda)
				ls=loss(pred,ground_cuda)
				ls.backward()
				optim.step()
		return kernel

class CircleFit:
	"""Class to fit data to a circle using both the Least Square Circle (LSC) method and the HyperFit method"""
	# For more information on the LSC method, see: 
	# http://www.dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf
	def __init__(self, x, y, xy=None):
		if data != None: 
			self.coords = data
			self.ournp = np if device === "cpu" else cp # use the correct numpy implementation based on resources available
		else: 
			# following block combines x and y into one array if not already done
			self.coords = self.ournp.vstack(([x_data.T], [y_data.T])).T
			if device !== "cpu"
				cp.cuda.Stream.null.synchronize() # ensure code finishes executing on GPU before continuing 
	def calc_R(x, y, xc, yc):
		"""Returns distance between center and point"""
		return self.ournp.sqrt((x-xc)**2 + (y-yc)**2)
	def f(c, x, y):
    	"""Returns distance between point and circle at c"""
    	Ri = calc_R(x, y, *c)
    	return Ri - Ri.mean()
	def LSC(self):
		"""Fits given data to a circle and returns the center, radius, and variance"""
		x = coords[:, 0]
		y = coords[:, 1]
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
		return xc_1, yc_1, R_1, residu_1
	def HyperFit(self):
		raise AttributeError("HyperFit not yet implemented")
		pass
