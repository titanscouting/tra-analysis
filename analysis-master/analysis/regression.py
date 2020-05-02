# Titan Robotics Team 2022: CUDA-based Regressions Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#   this module has been automatically inegrated into analysis.py, and should be callable as a class from the package
#   this module is cuda-optimized and vectorized (except for one small part)
# setup:

__version__ = "1.0.0.004"

# changelog should be viewed using print(analysis.regression.__changelog__)
__changelog__ = """
	1.0.0.004:
		- bug fixes
		- fixed changelog
	1.0.0.003:
		- bug fixes
	1.0.0.002:
		-Added more parameters to log, exponential, polynomial
		-Added SigmoidalRegKernelArthur, because Arthur apparently needs
		to train the scaling and shifting of sigmoids
	1.0.0.001:
		-initial release, with linear, log, exponential, polynomial, and sigmoid kernels
		-already vectorized (except for polynomial generation) and CUDA-optimized
"""

__author__ = (
	"Jacob Levine <jlevine@imsa.edu>",
	"Arthur Lu <learthurgo@gmail.com>"
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
	'CustomTrain'
]

import torch

global device

device = "cuda:0" if torch.torch.cuda.is_available() else "cpu"

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