# Titan Robotics Team 2022: Clustering submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import Clustering'
# setup:

__version__ = "2.0.2"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
	2.0.2:
		- generalized optional args to **kwargs
	2.0.1:
		- added normalization preprocessing to clustering, expects instance of sklearn.preprocessing.Normalizer()
	2.0.0:
		- added dbscan clustering algo
		- added spectral clustering algo
    1.0.0:
        - created this submodule
        - copied kmeans clustering from Analysis
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	"kmeans",
	"dbscan",
	"spectral",
]

import sklearn

def kmeans(data, normalizer = None, **kwargs):

	if  normalizer != None:
		data = normalizer.transform(data)

	kernel = sklearn.cluster.KMeans(**kwargs)
	kernel.fit(data)
	predictions = kernel.predict(data)
	centers = kernel.cluster_centers_

	return centers, predictions

def dbscan(data, normalizer=None, **kwargs):

	if  normalizer != None:
		data = normalizer.transform(data)

	model = sklearn.cluster.DBSCAN(**kwargs).fit(data)

	return model.labels_

def spectral(data, normalizer=None, **kwargs):

	if  normalizer != None:
		data = normalizer.transform(data)

	model = sklearn.cluster.SpectralClustering(**kwargs).fit(data)

	return model.labels_