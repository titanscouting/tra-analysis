
#Titan Robotics Team 2022: Data Analysis Script
#Written by Arthur Lu & Jacob Levine
#Notes:
#setup:

__version__ = "1.0.2.000"

__changelog__ = """changelog:
1.0.2.000:
	- added data reading from folder
	- nearly crashed computer reading from 20 GiB of data
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
import os
import glob

#get all the data
source_dir = 'data'
file_list = glob.glob(source_dir + '/*.CSV')
data = []
for file_path in file_list:
    data.append(analysis.load_csv(file_path))

#unhelpful comment
#for d in data: #unpacks 3d array into 2d
#    print (d)