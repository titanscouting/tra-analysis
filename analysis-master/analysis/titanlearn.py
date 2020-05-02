# Titan Robotics Team 2022: ML Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#    this should be imported as a python module using 'import titanlearn'
#    this should be included in the local directory or environment variable
#    this module is optimized for multhreaded computing
#    this module learns from its mistakes far faster than 2022's captains
# setup:

__version__ = "2.0.1.001"

#changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
	2.0.1.001:
		- removed matplotlib import
		- removed graphloss()
	2.0.1.000:
		- added net, dataset, dataloader, and stdtrain template definitions
		- added graphloss function
	2.0.0.001:
		- added clear functions
	2.0.0.000:
		- complete rewrite planned
		- depreciated 1.0.0.xxx versions
		- added simple training loop
	1.0.0.xxx:
		-added generation of ANNS, basic SGD training
"""

__author__ = (
	"Arthur Lu <arthurlu@ttic.edu>,"
	"Jacob Levine <jlevine@ttic.edu>,"
	)

__all__ = [
	'clear',
	'net',
	'dataset',
	'dataloader',
	'train',
	'stdtrainer',
	]

import torch
from os import system, name
import numpy as np

def clear(): 
	if name == 'nt': 
		_ = system('cls') 
	else: 
		_ = system('clear') 

class net(torch.nn.Module): #template for standard neural net
	def __init__(self):
		super(Net, self).__init__()
		
	def forward(self, input):
		pass

class dataset(torch.utils.data.Dataset): #template for standard dataset
	
	def __init__(self):
		super(torch.utils.data.Dataset).__init__()
		
	def __getitem__(self, index):
		pass
	
	def __len__(self):
		pass

def dataloader(dataset, batch_size, num_workers, shuffle = True):

	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def train(device, net, epochs, trainloader, optimizer, criterion): #expects standard dataloader, whch returns (inputs, labels)

	dataset_len = trainloader.dataset.__len__()
	iter_count = 0
	running_loss = 0
	running_loss_list = []

	for epoch in range(epochs):  # loop over the dataset multiple times

		for i, data in enumerate(trainloader, 0):

			inputs = data[0].to(device)
			labels = data[1].to(device)

			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, labels.to(torch.float))
			
			loss.backward()
			optimizer.step()

			# monitoring steps below

			iter_count += 1
			running_loss += loss.item()
			running_loss_list.append(running_loss)
			clear()

			print("training on: " + device)
			print("iteration: " + str(i) + "/" + str(int(dataset_len / trainloader.batch_size)) + " | " + "epoch: " + str(epoch) + "/" + str(epochs))
			print("current batch loss: " + str(loss.item))
			print("running loss: " + str(running_loss / iter_count))
		
	return net, running_loss_list
	print("finished training")

def stdtrainer(net, criterion, optimizer, dataloader, epochs, batch_size):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	net = net.to(device)
	criterion = criterion.to(device)
	optimizer = optimizer.to(device)
	trainloader = dataloader
	
	return train(device, net, epochs, trainloader, optimizer, criterion)