#Titan Robotics Team 2022: Data Analysis Script
#Written by Arthur Lu & Jacob Levine
#Notes:
#setup:

__version__ = "1.0.3.000"

__changelog__ = """changelog:
1.0.3.001:
        - processes data more efficiently
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

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import analysis
import titanlearn
import visualization
import os
import glob
import numpy as np

# Use a service account
cred = credentials.Certificate('keys/titanscoutandroid_firebase.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

#get all the data

analysis.generate_data("data/bdata.csv", 100, 5, -10, 10)

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

	measure_stats = []

	for i in range(len(measure)): #unpacks into specific teams

		ofbest_curve = [None]
		r2best_curve = [None]

		line = measure[i]

		#print(line)

		x = list(range(len(line)))
		eqs, rmss, r2s, overfit = analysis.optimize_regression(x, line, 10, 1)

		beqs, brmss, br2s, boverfit = analysis.select_best_regression(eqs, rmss, r2s, overfit, "min_overfit")

		#print(eqs, rmss, r2s, overfit)
		
		ofbest_curve.append(beqs)
		ofbest_curve.append(brmss)
		ofbest_curve.append(br2s)
		ofbest_curve.append(boverfit)
		ofbest_curve.pop(0)

		#print(ofbest_curve)

		beqs, brmss, br2s, boverfit = analysis.select_best_regression(eqs, rmss, r2s, overfit, "max_r2s")

		r2best_curve.append(beqs)
		r2best_curve.append(brmss)
		r2best_curve.append(br2s)
		r2best_curve.append(boverfit)
		r2best_curve.pop(0)

		#print(r2best_curve)
		
		measure_stats.append(teams[i] + ["|"] +  list(analysis.basic_stats(line, 0, 0)) + ["|"] + list(analysis.histo_analysis(line, 1, -3, 3)) + ["|"] + ofbest_curve + ["|"] + r2best_curve)

	stats.append(list(measure_stats))
	
json_out = {}
		
for i in range(len(stats)):
        json_out[files[i]]=stats[i]

db.collection(u'stats').document(u'stats-noNN').set(json_out)
