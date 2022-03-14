# Titan Robotics Team 2022: CorrelationTest submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import CorrelationTest'
# setup:

__version__ = "1.0.3"

__changelog__ = """changelog:
	1.0.3:
		- generalized optional args to **kwargs
	1.0.2:
		- optimized imports
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

def anova_oneway(*args): #expects arrays of samples

	results = scipy.stats.f_oneway(*args)
	return {"f-value": results[0], "p-value": results[1]}

def pearson(x, y):

	results = scipy.stats.pearsonr(x, y)
	return {"r-value": results[0], "p-value": results[1]}

def spearman(a, b = None, **kwargs):

	results = scipy.stats.spearmanr(a, b = b, **kwargs)
	return {"r-value": results[0], "p-value": results[1]}

def point_biserial(x, y):

	results = scipy.stats.pointbiserialr(x, y)
	return {"r-value": results[0], "p-value": results[1]}

def kendall(x, y, **kwargs):

	results = scipy.stats.kendalltau(x, y, **kwargs)
	return {"tau": results[0], "p-value": results[1]}

def kendall_weighted(x, y, **kwargs):

	results = scipy.stats.weightedtau(x, y, **kwargs)
	return {"tau": results[0], "p-value": results[1]}

def mgc(x, y, **kwargs):

	results = scipy.stats.multiscale_graphcorr(x, y, **kwargs)
	return {"k-value": results[0], "p-value": results[1], "data": results[2]} # unsure if MGC test returns a k value