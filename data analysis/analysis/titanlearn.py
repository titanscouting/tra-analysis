# Titan Robotics Team 2022: ML Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#    this should be imported as a python module using 'import titanlearn'
#    this should be included in the local directory or environment variable
#    this module is optimized for multhreaded computing
#    this module learns from its mistakes far faster than 2022's captains
# setup:

__version__ = "2.0.0.001"

#changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
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
    'train',
    ]

import torch
import torch.optim as optim
from os import *

def clear(): 
    if os.name == 'nt': 
        _ = os.system('cls') 
    else: 
        _ = os.system('clear') 

def train(device, net, epochs, trainloader, optimizer, criterion):

    for epoch in range(epochs):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):

            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels.to(torch.float))
            
            loss.backward()
            optimizer.step()
        
    return net
    print("finished training")