#Titan Robotics Team 2022: Data Analysis Script
#Written by Arthur Lu & Jacob Levine
#Notes:
#setup:

__version__ = "1.0.3.000"

__changelog__ = """changelog:
1.0.3.000:
	- actually processes data
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
file_list = glob.glob(source_dir + '/*.csv') #supposedly sorts by alphabetical order, skips reading teams.csv because of redundancy
data = []
files = [fn for fn in glob.glob('data/*.csv') 
         if not os.path.basename(fn).startswith('teams')]

#for file_path in file_list:
#	if not os.path.basename(file_list).startswith("teams")
#		data.append(analysis.load_csv(file_path))

for i in files:
	data.append(analysis.load_csv(i))

stats = []
measure_stats = []
teams = analysis.load_csv("data/teams.csv")

#assumes that team number is in the first column, and that the order of teams is the same across all files
#unhelpful comment
for measure in data: #unpacks 3d array into 2ds
	for i in range(len(measure)): #unpacks into specific teams
		line = measure[i]
		line.pop(0) #removes team identifier
		measure_stats.append(teams[i] + list(analysis.basic_stats(line, 0, 0)))
	stats.append(list(measure_stats))
		
print (stats)
#	print(d)

#print (stats)

