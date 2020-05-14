import data as d
import matplotlib.pyplot as plt
import numpy as np
import pymongo

def get_pit_variable_data(apikey, competition):
	client = pymongo.MongoClient(apikey)
	db = client.data_processing
	mdata = db.team_pit
	out = {}
	return mdata.find()

def get_pit_variable_formatted(apikey, competition):
	temp = get_pit_variable_data(apikey, competition)
	out = {}
	for i in temp:
		out[i["variable"]] = i["data"]
	return out

def main(apikey, competition):

	pit = get_pit_variable_formatted(apikey, competition)

	fig, ax = plt.subplots(1, len(pit), sharey=True, figsize=(80,15))

	i = 0

	for variable in pit:

		ax[i].hist(pit[variable])
		ax[i].invert_xaxis()

		ax[i].set_xlabel('')
		ax[i].set_ylabel('Frequency')
		ax[i].set_title(variable)

		plt.yticks(np.arange(len(pit[variable])))

		i+=1

	plt.show()