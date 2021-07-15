# Titan Robotics Team 2022: Clustering submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import Clustering'
# setup:

__version__ = "2.0.0"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
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

def kmeans(data, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=0.0001, precompute_distances="auto", verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm="auto"):

	kernel = sklearn.cluster.KMeans(n_clusters = n_clusters, init = init, n_init = n_init, max_iter = max_iter, tol = tol, precompute_distances = precompute_distances, verbose = verbose, random_state = random_state, copy_x = copy_x, n_jobs = n_jobs, algorithm = algorithm)
	kernel.fit(data)
	predictions = kernel.predict(data)
	centers = kernel.cluster_centers_

	return centers, predictions

def dbscan(data, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):

	model = sklearn.cluster.DBSCAN(eps = eps, min_samples = min_samples, metric = metric, metric_params = metric_params, algorithm = algorithm, leaf_size = leaf_size, p = p, n_jobs = n_jobs).fit(data)

	return model.labels_

def spectral(data, n_clusters=8, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):

	model = sklearn.cluster.SpectralClustering(n_clusters = n_clusters, eigen_solver = eigen_solver, n_components = n_components, random_state = random_state, n_init = n_init, gamma = gamma, affinity = affinity, n_neighbors = n_neighbors, eigen_tol = eigen_tol, assign_labels = assign_labels, degree = degree, coef0 = coef0, kernel_params = kernel_params, n_jobs = n_jobs).fit(data)

	return model.labels_