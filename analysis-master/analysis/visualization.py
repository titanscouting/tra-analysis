# Titan Robotics Team 2022: Visualization Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#    this should be imported as a python module using 'import visualization'
#    this should be included in the local directory or environment variable
#    fancy
# setup:

__version__ = "1.0.0.000"

#changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
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

def graphloss(losses):

	x = range(0, len(losses))
	plt.plot(x, losses)
	plt.show()