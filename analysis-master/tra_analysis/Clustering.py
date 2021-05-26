# Titan Robotics Team 2022: Clustering submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import Clustering'
# setup:

__version__ = "1.0.0"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
    1.0.0:
        - created this submodule
        - copied kmeans clustering from Analysis
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
]

def kmeans(data, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=0.0001, precompute_distances="auto", verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm="auto"):

	kernel = sklearn.cluster.KMeans(n_clusters = n_clusters, init = init, n_init = n_init, max_iter = max_iter, tol = tol, precompute_distances = precompute_distances, verbose = verbose, random_state = random_state, copy_x = copy_x, n_jobs = n_jobs, algorithm = algorithm)
	kernel.fit(data)
	predictions = kernel.predict(data)
	centers = kernel.cluster_centers_

	return centers, predictions