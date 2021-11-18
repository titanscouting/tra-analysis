# Titan Robotics Team 2022: Analysis Module
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import Analysis'
#    this should be included in the local directory or environment variable
#    this module has been optimized for multhreaded computing
#    current benchmark of optimization: 1.33 times faster
# setup:

__version__ = "3.0.6"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
	3.0.6:
		- added docstrings
	3.0.5:
		- removed extra submodule imports
		- fixed/optimized header
	3.0.4:
		- removed -_obj imports
	3.0.3:
		- fixed spelling of deprecate
	3.0.2:
		- fixed __all__
	3.0.1:
		- removed numba dependency and calls
	3.0.0:
		- exported several submodules to their own files while preserving backwards compatibility:
			- Array
			- ClassificationMetric
			- CorrelationTest
			- KNN
			- NaiveBayes
			- RandomForest
			- RegressionMetric
			- Sort
			- StatisticalTest
			- SVM
		- note: above listed submodules will not be supported in the future
		- future changes to all submodules will be held in their respective changelogs
		- future changes altering the parent package will be held in the __changelog__ of the parent package (in __init__.py)
		- changed reference to module name to Analysis
	2.3.1:
		- fixed bugs in Array class
	2.3.0:
		- overhauled Array class
	2.2.3:
		- fixed spelling of RandomForest
		- made n_neighbors required for KNN
		- made n_classifiers required for SVM
	2.2.2:
		- fixed 2.2.1 changelog entry
		- changed regression to return dictionary
	2.2.1:
		- changed all references to parent package analysis to tra_analysis
	2.2.0:
		- added Sort class
		- added several array sorting functions to Sort class including:
			- quick sort
			- merge sort
			- intro(spective) sort
			- heap sort
			- insertion sort
			- tim sort
			- selection sort
			- bubble sort
			- cycle sort
			- cocktail sort
		- tested all sorting algorithms with both lists and numpy arrays
		- deprecated sort function from Array class
		- added warnings as an import
	2.1.4:
		- added sort and search functions to Array class
	2.1.3:
		- changed output of basic_stats and histo_analysis to libraries
		- fixed __all__
	2.1.2:
		- renamed ArrayTest class to Array
	2.1.1:
		- added add, mul, neg, and inv functions to ArrayTest class
		- added normalize function to ArrayTest class
		- added dot and cross functions to ArrayTest class
	2.1.0:
		- added ArrayTest class
		- added elementwise mean, median, standard deviation, variance, min, max functions to ArrayTest class
		- added elementwise_stats to ArrayTest which encapsulates elementwise statistics
		- appended to __all__ to reflect changes
	2.0.6:
		- renamed func functions in regression to lin, log, exp, and sig 
	2.0.5:
		- moved random_forrest_regressor and random_forrest_classifier to RandomForrest class
		- renamed Metrics to Metric
		- renamed RegressionMetrics to RegressionMetric
		- renamed ClassificationMetrics to ClassificationMetric
		- renamed CorrelationTests to CorrelationTest
		- renamed StatisticalTests to StatisticalTest
		- reflected rafactoring to all mentions of above classes/functions
	2.0.4:
		- fixed __all__ to reflected the correct functions and classes
		- fixed CorrelationTests and StatisticalTests class functions to require self invocation
		- added missing math import
		- fixed KNN class functions to require self invocation
		- fixed Metrics class functions to require self invocation
		- various spelling fixes in CorrelationTests and StatisticalTests
	2.0.3:
		- bug fixes with CorrelationTests and StatisticalTests
		- moved glicko2 and trueskill to the metrics subpackage
		- moved elo to a new metrics subpackage
	2.0.2:
		- fixed docs
	2.0.1:
		- fixed docs
	2.0.0:
		- cleaned up wild card imports with scipy and sklearn
		- added CorrelationTests class
		- added StatisticalTests class
		- added several correlation tests to CorrelationTests
		- added several statistical tests to StatisticalTests
	1.13.9:
		- moved elo, glicko2, trueskill functions under class Metrics
	1.13.8:
		- moved Glicko2 to a seperate package
	1.13.7:
		- fixed bug with trueskill
	1.13.6:
		- cleaned up imports
	1.13.5:
		- cleaned up package
	1.13.4:
		- small fixes to regression to improve performance
	1.13.3:
		- filtered nans from regression
	1.13.2:
		- removed torch requirement, and moved Regression back to regression.py
	1.13.1:
		- bug fix with linear regression not returning a proper value
		- cleaned up regression
		- fixed bug with polynomial regressions
	1.13.0:
		- fixed all regressions to now properly work
	1.12.6:
		- fixed bg with a division by zero in histo_analysis
	1.12.5:
		- fixed numba issues by removing numba  from elo, glicko2 and trueskill
	1.12.4:
		- renamed gliko to glicko
	1.12.3:
		- removed deprecated code
	1.12.2:
		- removed team first time trueskill instantiation in favor of integration in superscript.py
	1.12.1:
		- improved readibility of regression outputs by stripping tensor data
		- used map with lambda to acheive the improved readibility
		- lost numba jit support with regression, and generated_jit hangs at execution
		- TODO: reimplement correct numba integration in regression
	1.12.0:
		- temporarily fixed polynomial regressions by using sklearn's PolynomialFeatures
	1.11.010:
		- alphabeticaly ordered import lists
	1.11.9:
		- bug fixes
	1.11.8:
		- bug fixes
	1.11.7:
		- bug fixes
	1.11.6:
		- tested min and max
		- bug fixes 
	1.11.5:
		- added min and max in basic_stats
	1.11.4:
		- bug fixes
	1.11.3:
		- bug fixes
	1.11.2:
		- consolidated metrics
		- fixed __all__
	1.11.1:
		- added test/train split to RandomForestClassifier and RandomForestRegressor
	1.11.0:
		- added RandomForestClassifier and RandomForestRegressor
		- note: untested
	1.10.0:
		- added numba.jit to remaining functions
	1.9.2:
		- kernelized PCA and KNN
	1.9.1:
		- fixed bugs with SVM and NaiveBayes
	1.9.0:
		- added SVM class, subclasses, and functions
		- note: untested
	1.8.0:
		- added NaiveBayes classification engine
		- note: untested
	1.7.0:
		- added knn()
		- added confusion matrix to decisiontree()
	1.6.2:
		- changed layout of __changelog to be vscode friendly
	1.6.1:
		- added additional hyperparameters to decisiontree()
	1.6.0:
		- fixed __version__
		- fixed __all__ order
		- added decisiontree()
	1.5.3:
		- added pca
	1.5.2:
		- reduced import list
		- added kmeans clustering engine
	1.5.1:
		- simplified regression by using .to(device)
	1.5.0:
		- added polynomial regression to regression(); untested
	1.4.0:
		- added trueskill()
	1.3.2:
		- renamed regression class to Regression, regression_engine() to regression gliko2_engine class to Gliko2
	1.3.1:
		- changed glicko2() to return tuple instead of array
	1.3.0:
		- added glicko2_engine class and glicko()
		- verified glicko2() accuracy
	1.2.3:
		- fixed elo()
	1.2.2:
		- added elo()
		- elo() has bugs to be fixed
	1.2.1:
		- readded regrression import
	1.2.0:
		- integrated regression.py as regression class
		- removed regression import
		- fixed metadata for regression class
		- fixed metadata for analysis class
	1.1.1:
		- regression_engine() bug fixes, now actaully regresses
	1.1.0:
		- added regression_engine()
		- added all regressions except polynomial
	1.0.7:
		- updated _init_device()
	1.0.6:
		- removed useless try statements
	1.0.5:
		- removed impossible outcomes
	1.0.4:
		- added performance metrics (r^2, mse, rms)
	1.0.3:
		- resolved nopython mode for mean, median, stdev, variance
	1.0.2:
		- snapped (removed) majority of uneeded imports
		- forced object mode (bad) on all jit
		- TODO: stop numba complaining about not being able to compile in nopython mode
	1.0.1:
		- removed from sklearn import * to resolve uneeded wildcard imports
	1.0.0:
		- removed c_entities,nc_entities,obstacles,objectives from __all__
		- applied numba.jit to all functions
		- deprecated and removed stdev_z_split
		- cleaned up histo_analysis to include numpy and numba.jit optimizations
		- deprecated and removed all regression functions in favor of future pytorch optimizer
		- deprecated and removed all nonessential functions (basic_analysis, benchmark, strip_data)
		- optimized z_normalize using sklearn.preprocessing.normalize
		- TODO: implement kernel/function based pytorch regression optimizer
	0.9.0:
		- refactored
		- numpyed everything
		- removed stats in favor of numpy functions
	0.8.5:
		- minor fixes
	0.8.4:
		- removed a few unused dependencies
	0.8.3:
		- added p_value function
	0.8.2:
		- updated __all__ correctly to contain changes made in v 0.8.0 and v 0.8.1
	0.8.1:
		- refactors
		- bugfixes
	0.8.0:
		- deprecated histo_analysis_old
		- deprecated debug
		- altered basic_analysis to take array data instead of filepath
		- refactor
		- optimization
	0.7.2:
		- bug fixes
	0.7.1:
		- bug fixes
	0.7.0:
		- added tanh_regression (logistical regression)
		- bug fixes
	0.6.5:
		- added z_normalize function to normalize dataset
		- bug fixes
	0.6.4:
		- bug fixes
	0.6.3:
		- bug fixes
	0.6.2:
		- bug fixes
	0.6.1:
		- corrected __all__ to contain all of the functions
	0.6.0:
		- added calc_overfit, which calculates two measures of overfit, error and performance
		- added calculating overfit to optimize_regression
	0.5.0:
		- added optimize_regression function, which is a sample function to find the optimal regressions
		- optimize_regression function filters out some overfit funtions (functions with r^2 = 1)
		- planned addition: overfit detection in the optimize_regression function
	0.4.2:
		- added __changelog__
		- updated debug function with log and exponential regressions
	0.4.1:
		- added log regressions
		- added exponential regressions
		- added log_regression and exp_regression to __all__
	0.3.8:
		- added debug function to further consolidate functions
	0.3.7:
		- added builtin benchmark function
		- added builtin random (linear) data generation function
		- added device initialization (_init_device)
	0.3.6:
		- reorganized the imports list to be in alphabetical order
		- added search and regurgitate functions to c_entities, nc_entities, obstacles, objectives
	0.3.5:
		- major bug fixes
		- updated historical analysis
		- deprecated old historical analysis
	0.3.4:
		- added __version__, __author__, __all__
		- added polynomial regression
		- added root mean squared function
		- added r squared function
	0.3.3:
		- bug fixes
		- added c_entities
	0.3.2:
		- bug fixes
		- added nc_entities, obstacles, objectives
		- consolidated statistics.py to analysis.py
	0.3.1:
		- compiled 1d, column, and row basic stats into basic stats function
	0.3.0:
		- added historical analysis function
	0.2.x:
		- added z score test
	0.1.x:
		- major bug fixes
	0.0.x:
		- added loading csv
		- added 1d, column, row basic stats
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	'load_csv',
	'basic_stats',
	'z_score',
	'z_normalize',
	'histo_analysis',
	'regression',
	'Metric',
	'pca',
	'decisiontree',
	# all statistics functions left out due to integration in other functions
]

# now back to your regularly scheduled programming:

# imports (now in alphabetical order! v 0.3.006):

import csv
from tra_analysis.metrics import elo as Elo
from tra_analysis.metrics import glicko2 as Glicko2
import numpy as np
import scipy
import sklearn, sklearn.cluster, sklearn.pipeline
from tra_analysis.metrics import trueskill as Trueskill

# import submodules

from .ClassificationMetric import ClassificationMetric

class error(ValueError):
	pass

def load_csv(filepath):
	"""
	Loads csv file into 2D numpy array. Does not check csv file validity.
	parameters:
		filepath: String path to the csv file
	return:
		2D numpy array of values stored in csv file
	"""
	with open(filepath, newline='') as csvfile:
		file_array = np.array(list(csv.reader(csvfile)))
		csvfile.close()
	return file_array

def basic_stats(data):
	"""
	Calculates mean, median, standard deviation, variance, minimum, maximum of a simple set of elements.
	parameters:
		data: List representing set of unordered elements
	return:
		Dictionary with (mean, median, standard-deviation, variance, minimum, maximum) as keys and corresponding values
	"""
	data_t = np.array(data).astype(float)

	_mean = mean(data_t)
	_median = median(data_t)
	_stdev = stdev(data_t)
	_variance = variance(data_t)
	_min = npmin(data_t)
	_max = npmax(data_t)

	return {"mean": _mean, "median": _median, "standard-deviation": _stdev, "variance": _variance, "minimum": _min, "maximum": _max}

def z_score(point, mean, stdev):
	"""
	Calculates z score of a specific point given mean and standard deviation of data.
	parameters:
		point: Real value corresponding to a single point of data
		mean: Real value corresponding to the mean of the dataset
		stdev: Real value corresponding to the standard deviation of the dataset
	return:
		Real value that is the point's z score
	"""
	score = (point - mean) / stdev
	
	return score

def z_normalize(array, *args):
	"""
	Applies sklearn.normalize(array, axis = args) on any arraylike parseable by numpy.
	parameters:
		array: array like structure of reals aka nested indexables
		*args: arguments relating to axis normalized against
	return:
		numpy array of normalized values from ArrayLike input
	"""
	array = np.array(array)
	for arg in args:
		array = sklearn.preprocessing.normalize(array, axis = arg)

	return array

def histo_analysis(hist_data):
	"""
	Calculates the mean and standard deviation of derivatives of (x,y) points. Requires at least 2 points to compute.
	parameters:
		hist_data: list of real coordinate point data (x, y)
	return:
		Dictionary with (mean, deviation) as keys to corresponding values
	"""
	if len(hist_data[0]) > 2:

		hist_data = np.array(hist_data)
		derivative = np.array(len(hist_data) - 1, dtype = float)
		t = np.diff(hist_data)
		derivative = t[1] / t[0]
		np.sort(derivative)

		return {"mean": basic_stats(derivative)["mean"], "deviation": basic_stats(derivative)["standard-deviation"]}

	else:

		return None

def regression(inputs, outputs, args): # inputs, outputs expects N-D array 
	"""
	Applies specified regression kernels onto input, output data pairs.
	parameters:
		inputs: List of Reals representing independent variable values of each point
		outputs: List of Reals representing dependent variable values of each point
		args: List of Strings from values (lin, log, exp, ply, sig)
	return:
		Dictionary with keys (lin, log, exp, ply, sig) as keys to correspondiong regression models
	"""
	X = np.array(inputs)
	y = np.array(outputs)

	regressions = {}

	if 'lin' in args: # formula: ax + b

		try:

			def lin(x, a, b):

				return a * x + b

			popt, pcov = scipy.optimize.curve_fit(lin, X, y)

			coeffs = popt.flatten().tolist()
			regressions["lin"] = (str(coeffs[0]) + "*x+" + str(coeffs[1]))

		except Exception as e:

			pass

	if 'log' in args: # formula: a log (b(x + c)) + d

		try:

			def log(x, a, b, c, d):

				return a * np.log(b*(x + c)) + d

			popt, pcov = scipy.optimize.curve_fit(log, X, y)

			coeffs = popt.flatten().tolist()
			regressions["log"] = (str(coeffs[0]) + "*log(" + str(coeffs[1]) + "*(x+" + str(coeffs[2]) + "))+" + str(coeffs[3]))

		except Exception as e:
			
			pass

	if 'exp' in args: # formula: a e ^ (b(x + c)) + d

		try:        

			def exp(x, a, b, c, d):

				return a * np.exp(b*(x + c)) + d

			popt, pcov = scipy.optimize.curve_fit(exp, X, y)

			coeffs = popt.flatten().tolist()
			regressions["exp"] = (str(coeffs[0]) + "*e^(" + str(coeffs[1]) + "*(x+" + str(coeffs[2]) + "))+" + str(coeffs[3]))

		except Exception as e:

			pass

	if 'ply' in args: # formula: a + bx^1 + cx^2 + dx^3 + ...
		
		inputs = np.array([inputs])
		outputs = np.array([outputs])

		plys = {}
		limit = len(outputs[0])

		for i in range(2, limit):

			model = sklearn.preprocessing.PolynomialFeatures(degree = i)
			model = sklearn.pipeline.make_pipeline(model, sklearn.linear_model.LinearRegression())
			model = model.fit(np.rot90(inputs), np.rot90(outputs))

			params = model.steps[1][1].intercept_.tolist()
			params = np.append(params, model.steps[1][1].coef_[0].tolist()[1::])
			params = params.flatten().tolist()

			temp = ""
			counter = 0
			for param in params:
				temp += "(" + str(param) + "*x^" + str(counter) + ")"
				counter += 1
			plys["x^" + str(i)] = (temp)

		regressions["ply"] = (plys)

	if 'sig' in args: # formula: a tanh (b(x + c)) + d

		try:        

			def sig(x, a, b, c, d):

				return a * np.tanh(b*(x + c)) + d

			popt, pcov = scipy.optimize.curve_fit(sig, X, y)

			coeffs = popt.flatten().tolist()
			regressions["sig"] = (str(coeffs[0]) + "*tanh(" + str(coeffs[1]) + "*(x+" + str(coeffs[2]) + "))+" + str(coeffs[3]))

		except Exception as e:
		   
			pass

	return regressions

class Metric:
	"""
	The metric class wraps the metrics models. Call without instantiation as Metric.<method>(...)
	"""
	def elo(self, starting_score, opposing_score, observed, N, K):
		"""
		Calculates an elo adjusted ELO score given a player's current score, opponent's score, and outcome of match.
		reference: https://en.wikipedia.org/wiki/Elo_rating_system
		parameters:
			starting_score: Real value representing player's ELO score before a match
			opposing_score: Real value representing opponent's score before the match
			observed: Array of Real values representing multiple sequential match outcomes against the same opponent. 1 for match win, 0.5 for tie, 0 for loss.
			N: Real value representing the normal or mean score expected (usually 1200)
			K: R eal value representing a system constant, determines how quickly players will change scores (usually 24)
		return:
			Real value representing the player's new ELO score
		"""
		return Elo.calculate(starting_score, opposing_score, observed, N, K)

	def glicko2(self, starting_score, starting_rd, starting_vol, opposing_score, opposing_rd, observations):
		"""
		Calculates an adjusted Glicko-2 score given a player's current score, multiple opponent's score, and outcome of several matches.
		reference: http://www.glicko.net/glicko/glicko2.pdf
		parameters:
			starting_score: Real value representing the player's Glicko-2 score
			starting_rd: Real value representing the player's RD
			starting_vol: Real value representing the player's volatility
			opposing_score: List of Real values representing multiple opponent's Glicko-2 scores
			opposing_rd: List of Real values representing multiple opponent's RD
			opposing_vol: List of Real values representing multiple opponent's volatility
			observations: List of Real values representing the outcome of several matches, where each match's opponent corresponds with the opposing_score, opposing_rd, opposing_vol values of the same indesx. Outcomes can be a score, presuming greater score is better. 
		return:
			Tuple of 3 Real values representing the player's new score, rd, and vol
		"""
		player = Glicko2.Glicko2(rating = starting_score, rd = starting_rd, vol = starting_vol)

		player.update_player([x for x in opposing_score], [x for x in opposing_rd], observations)

		return (player.rating, player.rd, player.vol)

	def trueskill(self, teams_data, observations): # teams_data is array of array of tuples ie. [[(mu, sigma), (mu, sigma), (mu, sigma)], [(mu, sigma), (mu, sigma), (mu, sigma)]]
		"""
		Calculates the score changes for multiple teams playing in a single match accoding to the trueskill algorithm.
		reference: https://trueskill.org/
		parameters:
			teams_data: List of List of Tuples of 2 Real values representing multiple player ratings. List of teams, which is a List of players. Each player rating is a Tuple of 2 Real values (mu, sigma).
			observations: List of Real values representing the match outcome. Each value in the List is the score corresponding to the team at the same index in teams_data.
		return:
			List of List of Tuples of 2 Real values representing new player ratings. Same structure as teams_data. 
		"""
		team_ratings = []

		for team in teams_data:
			team_temp = ()
			for player in team:
				player = Trueskill.Rating(player[0], player[1])
				team_temp = team_temp + (player,)
			team_ratings.append(team_temp)

		return Trueskill.rate(team_ratings, ranks=observations)

def mean(data):

	return np.mean(data)

def median(data):

	return np.median(data)

def stdev(data):

	return np.std(data)

def variance(data):

	return np.var(data)

def npmin(data):

	return np.amin(data)

def npmax(data):

	return np.amax(data)

def pca(data, n_components = None, copy = True, whiten = False, svd_solver = "auto", tol = 0.0, iterated_power = "auto", random_state = None):
	"""
	Performs a principle component analysis on the input data.
	reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	parameters:
		data: Arraylike of Reals representing the set of data to perform PCA on
		* : refer to reference for usage, parameters follow same usage
	return:
		Arraylike of Reals representing the set of data that has had PCA performed. The dimensionality of the Arraylike may be smaller or equal. 
	"""
	kernel = sklearn.decomposition.PCA(n_components = n_components, copy = copy, whiten = whiten, svd_solver = svd_solver, tol = tol, iterated_power = iterated_power, random_state = random_state)

	return kernel.fit_transform(data)

def decisiontree(data, labels, test_size = 0.3, criterion = "gini", splitter = "default", max_depth = None): #expects *2d data and 1d labels
	"""
	Generates a decision tree classifier fitted to the given data. 
	reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
	parameters:
		data: List of values representing each data point of multiple axes
		labels: List of values represeing the labels corresponding to the same index at data
		* : refer to reference for usage, parameters follow same usage
	return:
		DecisionTreeClassifier model and corresponding classification accuracy metrics
	"""
	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.tree.DecisionTreeClassifier(criterion = criterion, splitter = splitter, max_depth = max_depth)
	model = model.fit(data_train,labels_train)
	predictions = model.predict(data_test)
	metrics = ClassificationMetric(predictions, labels_test)

	return model, metrics