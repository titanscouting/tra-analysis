
#Titan Robotics Team 2022: Data Analysis Module
#Written by Arthur Lu & Jacob Levine
#Notes:
#setup:

__version__ = "1.0.1.000"

#changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
1.0.1.000:
	- added data reading from file
	- added superstructure to code
1.0.0.000:
	- added import statements (revolutionary)
""" 

__author__ = (
    "Arthur Lu <arthurlu@ttic.edu>, "
    "Jacob Levine <jlevine@ttic.edu>,"
    )

import analysis
import titanlearn
import visualization

data = analysis.load_csv("data/data.csv")