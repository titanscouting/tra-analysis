# Titan Robotics Team 2022: CorrelationTest submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import CorrelationTest'
# setup:

__version__ = "1.0.1"

__changelog__ = """changelog:
	1.0.1:
		- fixed __all__
	1.0.0:
		- ported analysis.CorrelationTest() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	"anova_oneway",
	"pearson",
	"spearman",
	"point_biserial",
	"kendall",
	"kendall_weighted",
	"mgc",
]

import scipy
from scipy import stats

def anova_oneway(*args): #expects arrays of samples

	results = scipy.stats.f_oneway(*args)
	return {"f-value": results[0], "p-value": results[1]}

def pearson(x, y):

	results = scipy.stats.pearsonr(x, y)
	return {"r-value": results[0], "p-value": results[1]}

def spearman(a, b = None, axis = 0, nan_policy = 'propagate'):

	results = scipy.stats.spearmanr(a, b = b, axis = axis, nan_policy = nan_policy)
	return {"r-value": results[0], "p-value": results[1]}

def point_biserial(x, y):

	results = scipy.stats.pointbiserialr(x, y)
	return {"r-value": results[0], "p-value": results[1]}

def kendall(x, y, initial_lexsort = None, nan_policy = 'propagate', method = 'auto'):

	results = scipy.stats.kendalltau(x, y, initial_lexsort = initial_lexsort, nan_policy = nan_policy, method = method)
	return {"tau": results[0], "p-value": results[1]}

def kendall_weighted(x, y, rank = True, weigher = None, additive = True):

	results = scipy.stats.weightedtau(x, y, rank = rank, weigher = weigher, additive = additive)
	return {"tau": results[0], "p-value": results[1]}

def mgc(x, y, compute_distance = None, reps = 1000, workers = 1, is_twosamp = False, random_state = None):

	results = scipy.stats.multiscale_graphcorr(x, y, compute_distance = compute_distance, reps = reps, workers = workers, is_twosamp = is_twosamp, random_state = random_state)
	return {"k-value": results[0], "p-value": results[1], "data": results[2]} # unsure if MGC test returns a k value