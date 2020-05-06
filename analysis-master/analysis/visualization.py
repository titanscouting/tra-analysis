# Titan Robotics Team 2022: Visualization Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#    this should be imported as a python module using 'import visualization'
#    this should be included in the local directory or environment variable
#    fancy
# setup:

__version__ = "1.0.0.001"

#changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
	1.0.0.001:
		- added graphhistogram function as a fragment of visualize_pit.py
	1.0.0.000:
		- created visualization.py
		- added graphloss()
		- added imports
"""

__author__ = (
	"Arthur Lu <arthurlu@ttic.edu>,"
	"Jacob Levine <jlevine@ttic.edu>,"
	)

__all__ = [
	'graphloss',
	]

import matplotlib.pyplot as plt
import numpy as np

def graphloss(losses):

	x = range(0, len(losses))
	plt.plot(x, losses)
	plt.show()

def graphhistogram(data, figsize, sharey = True): # expects library with key as variable and contents as occurances

	fig, ax = plt.subplots(1, len(data), sharey=sharey, figsize=figsize)

	i = 0

	for variable in data:

		ax[i].hist(data[variable])
		ax[i].invert_xaxis()

		ax[i].set_xlabel('Variable')
		ax[i].set_ylabel('Frequency')
		ax[i].set_title(variable)

		plt.yticks(np.arange(len(data[variable])))

		i+=1

	plt.show()