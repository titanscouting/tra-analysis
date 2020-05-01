# Titan Robotics Team 2022: Data Analysis Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#    this should be imported as a python module using 'from analysis import analysis'
#    this should be included in the local directory or environment variable
#    this module has been optimized for multhreaded computing
#    current benchmark of optimization: 1.33 times faster
# setup:

__version__ = "1.2.0.004"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
	1.2.0.004:
		- fixed __all__ to reflected the correct functions and classes
		- fixed CorrelationTests and StatisticalTests class functions to require self invocation
		- added missing math import
		- fixed KNN class functions to require self invocation
		- fixed Metrics class functions to require self invocation
		- various spelling fixes in CorrelationTests and StatisticalTests
	1.2.0.003:
		- bug fixes with CorrelationTests and StatisticalTests
		- moved glicko2 and trueskill to the metrics subpackage
		- moved elo to a new metrics subpackage
	1.2.0.002:
		- fixed docs
	1.2.0.001:
		- fixed docs
	1.2.0.000:
		- cleaned up wild card imports with scipy and sklearn
		- added CorrelationTests class
		- added StatisticalTests class
		- added several correlation tests to CorrelationTests
		- added several statistical tests to StatisticalTests
	1.1.13.009:
		- moved elo, glicko2, trueskill functions under class Metrics
	1.1.13.008:
		- moved Glicko2 to a seperate package
	1.1.13.007:
		- fixed bug with trueskill
	1.1.13.006:
		- cleaned up imports
	1.1.13.005:
		- cleaned up package
	1.1.13.004:
		- small fixes to regression to improve performance
	1.1.13.003:
		- filtered nans from regression
	1.1.13.002:
		- removed torch requirement, and moved Regression back to regression.py
	1.1.13.001:
		- bug fix with linear regression not returning a proper value
		- cleaned up regression
		- fixed bug with polynomial regressions
	1.1.13.000:
		- fixed all regressions to now properly work
	1.1.12.006:
		- fixed bg with a division by zero in histo_analysis
	1.1.12.005:
		- fixed numba issues by removing numba  from elo, glicko2 and trueskill
	1.1.12.004:
		- renamed gliko to glicko
	1.1.12.003:
		- removed depreciated code
	1.1.12.002:
		- removed team first time trueskill instantiation in favor of integration in superscript.py
	1.1.12.001:
		- improved readibility of regression outputs by stripping tensor data
		- used map with lambda to acheive the improved readibility
		- lost numba jit support with regression, and generated_jit hangs at execution
		- TODO: reimplement correct numba integration in regression
	1.1.12.000:
		- temporarily fixed polynomial regressions by using sklearn's PolynomialFeatures
	1.1.11.010:
		- alphabeticaly ordered import lists
	1.1.11.009:
		- bug fixes
	1.1.11.008:
		- bug fixes
	1.1.11.007:
		- bug fixes
	1.1.11.006:
		- tested min and max
		- bug fixes 
	1.1.11.005:
		- added min and max in basic_stats
	1.1.11.004:
		- bug fixes
	1.1.11.003:
		- bug fixes
	1.1.11.002:
		- consolidated metrics
		- fixed __all__
	1.1.11.001:
		- added test/train split to RandomForestClassifier and RandomForestRegressor
	1.1.11.000:
		- added RandomForestClassifier and RandomForestRegressor
		- note: untested
	1.1.10.000:
		- added numba.jit to remaining functions
	1.1.9.002:
		- kernelized PCA and KNN
	1.1.9.001:
		- fixed bugs with SVM and NaiveBayes
	1.1.9.000:
		- added SVM class, subclasses, and functions
		- note: untested
	1.1.8.000:
		- added NaiveBayes classification engine
		- note: untested
	1.1.7.000:
		- added knn()
		- added confusion matrix to decisiontree()
	1.1.6.002:
		- changed layout of __changelog to be vscode friendly
	1.1.6.001:
		- added additional hyperparameters to decisiontree()
	1.1.6.000:
		- fixed __version__
		- fixed __all__ order
		- added decisiontree()
	1.1.5.003:
		- added pca
	1.1.5.002:
		- reduced import list
		- added kmeans clustering engine
	1.1.5.001:
		- simplified regression by using .to(device)
	1.1.5.000:
		- added polynomial regression to regression(); untested
	1.1.4.000:
		- added trueskill()
	1.1.3.002:
		- renamed regression class to Regression, regression_engine() to regression gliko2_engine class to Gliko2
	1.1.3.001:
		- changed glicko2() to return tuple instead of array
	1.1.3.000:
		- added glicko2_engine class and glicko()
		- verified glicko2() accuracy
	1.1.2.003:
		- fixed elo()
	1.1.2.002:
		- added elo()
		- elo() has bugs to be fixed
	1.1.2.001:
		- readded regrression import
	1.1.2.000:
		- integrated regression.py as regression class
		- removed regression import
		- fixed metadata for regression class
		- fixed metadata for analysis class
	1.1.1.001:
		- regression_engine() bug fixes, now actaully regresses
	1.1.1.000:
		- added regression_engine()
		- added all regressions except polynomial
	1.1.0.007:
		- updated _init_device()
	1.1.0.006:
		- removed useless try statements
	1.1.0.005:
		- removed impossible outcomes
	1.1.0.004:
		- added performance metrics (r^2, mse, rms)
	1.1.0.003:
		- resolved nopython mode for mean, median, stdev, variance
	1.1.0.002:
		- snapped (removed) majority of uneeded imports
		- forced object mode (bad) on all jit
		- TODO: stop numba complaining about not being able to compile in nopython mode
	1.1.0.001:
		- removed from sklearn import * to resolve uneeded wildcard imports
	1.1.0.000:
		- removed c_entities,nc_entities,obstacles,objectives from __all__
		- applied numba.jit to all functions
		- depreciated and removed stdev_z_split
		- cleaned up histo_analysis to include numpy and numba.jit optimizations
		- depreciated and removed all regression functions in favor of future pytorch optimizer
		- depreciated and removed all nonessential functions (basic_analysis, benchmark, strip_data)
		- optimized z_normalize using sklearn.preprocessing.normalize
		- TODO: implement kernel/function based pytorch regression optimizer
	1.0.9.000:
		- refactored
		- numpyed everything
		- removed stats in favor of numpy functions
	1.0.8.005:
		- minor fixes
	1.0.8.004:
		- removed a few unused dependencies
	1.0.8.003:
		- added p_value function
	1.0.8.002:
		- updated __all__ correctly to contain changes made in v 1.0.8.000 and v 1.0.8.001
	1.0.8.001:
		- refactors
		- bugfixes
	1.0.8.000:
		- depreciated histo_analysis_old
		- depreciated debug
		- altered basic_analysis to take array data instead of filepath
		- refactor
		- optimization
	1.0.7.002:
		- bug fixes
	1.0.7.001:
		- bug fixes
	1.0.7.000:
		- added tanh_regression (logistical regression)
		- bug fixes
	1.0.6.005:
		- added z_normalize function to normalize dataset
		- bug fixes
	1.0.6.004:
		- bug fixes
	1.0.6.003:
		- bug fixes
	1.0.6.002:
		- bug fixes
	1.0.6.001:
		- corrected __all__ to contain all of the functions
	1.0.6.000:
		- added calc_overfit, which calculates two measures of overfit, error and performance
		- added calculating overfit to optimize_regression
	1.0.5.000:
		- added optimize_regression function, which is a sample function to find the optimal regressions
		- optimize_regression function filters out some overfit funtions (functions with r^2 = 1)
		- planned addition: overfit detection in the optimize_regression function
	1.0.4.002:
		- added __changelog__
		- updated debug function with log and exponential regressions
	1.0.4.001:
		- added log regressions
		- added exponential regressions
		- added log_regression and exp_regression to __all__
	1.0.3.008:
		- added debug function to further consolidate functions
	1.0.3.007:
		- added builtin benchmark function
		- added builtin random (linear) data generation function
		- added device initialization (_init_device)
	1.0.3.006:
		- reorganized the imports list to be in alphabetical order
		- added search and regurgitate functions to c_entities, nc_entities, obstacles, objectives
	1.0.3.005:
		- major bug fixes
		- updated historical analysis
		- depreciated old historical analysis
	1.0.3.004:
		- added __version__, __author__, __all__
		- added polynomial regression
		- added root mean squared function
		- added r squared function
	1.0.3.003:
		- bug fixes
		- added c_entities
	1.0.3.002:
		- bug fixes
		- added nc_entities, obstacles, objectives
		- consolidated statistics.py to analysis.py
	1.0.3.001:
		- compiled 1d, column, and row basic stats into basic stats function
	1.0.3.000:
		- added historical analysis function
	1.0.2.xxx:
		- added z score test
	1.0.1.xxx:
		- major bug fixes
	1.0.0.xxx:
		- added loading csv
		- added 1d, column, row basic stats
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
	"Jacob Levine <jlevine@imsa.edu>",
)

__all__ = [
	'load_csv',
	'basic_stats',
	'z_score',
	'z_normalize',
	'histo_analysis',
	'regression',
	'Metrics',
	'RegressionMetrics',
	'ClassificationMetrics',
	'kmeans',
	'pca',
	'decisiontree',
	'KNN',
	'NaiveBayes',
	'SVM',
	'random_forest_classifier',
	'random_forest_regressor',
	'CorrelationTests',
	'StatisticalTests',
	# all statistics functions left out due to integration in other functions
]

# now back to your regularly scheduled programming:

# imports (now in alphabetical order! v 1.0.3.006):

import csv
from analysis.metrics import elo as Elo
from analysis.metrics import glicko2 as Glicko2
import math
import numba
from numba import jit
import numpy as np
import scipy
from scipy import optimize, stats
import sklearn
from sklearn import preprocessing, pipeline, linear_model, metrics, cluster, decomposition, tree, neighbors, naive_bayes, svm, model_selection, ensemble
from analysis.metrics import trueskill as Trueskill

class error(ValueError):
	pass

def load_csv(filepath):
	with open(filepath, newline='') as csvfile:
		file_array = np.array(list(csv.reader(csvfile)))
		csvfile.close()
	return file_array

# expects 1d array
@jit(forceobj=True)
def basic_stats(data):

	data_t = np.array(data).astype(float)

	_mean = mean(data_t)
	_median = median(data_t)
	_stdev = stdev(data_t)
	_variance = variance(data_t)
	_min = npmin(data_t)
	_max = npmax(data_t)

	return _mean, _median, _stdev, _variance, _min, _max

# returns z score with inputs of point, mean and standard deviation of spread
@jit(forceobj=True)
def z_score(point, mean, stdev):
	score = (point - mean) / stdev
	
	return score

# expects 2d array, normalizes across all axes
@jit(forceobj=True)
def z_normalize(array, *args):

   array = np.array(array)
   for arg in args:
	   array = sklearn.preprocessing.normalize(array, axis = arg)

   return array

@jit(forceobj=True)
# expects 2d array of [x,y]
def histo_analysis(hist_data):

	if(len(hist_data[0]) > 2):

		hist_data = np.array(hist_data)
		derivative = np.array(len(hist_data) - 1, dtype = float)
		t = np.diff(hist_data)
		derivative = t[1] / t[0]
		np.sort(derivative)

		return basic_stats(derivative)[0], basic_stats(derivative)[3]

	else:

		return None

def regression(inputs, outputs, args): # inputs, outputs expects N-D array 

	X = np.array(inputs)
	y = np.array(outputs)

	regressions = []

	if 'lin' in args: # formula: ax + b

		try:

			def func(x, a, b):

				return a * x + b

			popt, pcov = scipy.optimize.curve_fit(func, X, y)

			regressions.append((popt.flatten().tolist(), None))

		except Exception as e:

			pass

	if 'log' in args: # formula: a log (b(x + c)) + d

		try:

			def func(x, a, b, c, d):

				return a * np.log(b*(x + c)) + d

			popt, pcov = scipy.optimize.curve_fit(func, X, y)

			regressions.append((popt.flatten().tolist(), None))

		except Exception as e:
			
			pass

	if 'exp' in args: # formula: a e ^ (b(x + c)) + d

		try:        

			def func(x, a, b, c, d):

				return a * np.exp(b*(x + c)) + d

			popt, pcov = scipy.optimize.curve_fit(func, X, y)

			regressions.append((popt.flatten().tolist(), None))

		except Exception as e:

			pass

	if 'ply' in args: # formula: a + bx^1 + cx^2 + dx^3 + ...
		
		inputs = np.array([inputs])
		outputs = np.array([outputs])

		plys = []
		limit = len(outputs[0])

		for i in range(2, limit):

			model = sklearn.preprocessing.PolynomialFeatures(degree = i)
			model = sklearn.pipeline.make_pipeline(model, sklearn.linear_model.LinearRegression())
			model = model.fit(np.rot90(inputs), np.rot90(outputs))

			params = model.steps[1][1].intercept_.tolist()
			params = np.append(params, model.steps[1][1].coef_[0].tolist()[1::])
			params.flatten()
			params = params.tolist()
			
			plys.append(params)

		regressions.append(plys)

	if 'sig' in args: # formula: a tanh (b(x + c)) + d

		try:        

			def func(x, a, b, c, d):

				return a * np.tanh(b*(x + c)) + d

			popt, pcov = scipy.optimize.curve_fit(func, X, y)

			regressions.append((popt.flatten().tolist(), None))

		except Exception as e:
		   
			pass

	return regressions

class Metrics:

	def elo(self, starting_score, opposing_score, observed, N, K):

		return Elo.calculate(starting_score, opposing_score, observed, N, K)

	def glicko2(self, starting_score, starting_rd, starting_vol, opposing_score, opposing_rd, observations):

		player = Glicko2.Glicko2(rating = starting_score, rd = starting_rd, vol = starting_vol)

		player.update_player([x for x in opposing_score], [x for x in opposing_rd], observations)

		return (player.rating, player.rd, player.vol)

	def trueskill(self, teams_data, observations): # teams_data is array of array of tuples ie. [[(mu, sigma), (mu, sigma), (mu, sigma)], [(mu, sigma), (mu, sigma), (mu, sigma)]]

		team_ratings = []

		for team in teams_data:
			team_temp = ()
			for player in team:
				player = Trueskill.Rating(player[0], player[1])
				team_temp = team_temp + (player,)
			team_ratings.append(team_temp)

		return Trueskill.rate(team_ratings, ranks=observations)

class RegressionMetrics():

	def __new__(cls, predictions, targets):

		return cls.r_squared(cls, predictions, targets), cls.mse(cls, predictions, targets), cls.rms(cls, predictions, targets)

	def r_squared(self, predictions, targets):  # assumes equal size inputs

		return sklearn.metrics.r2_score(targets, predictions)

	def mse(self, predictions, targets):

		return sklearn.metrics.mean_squared_error(targets, predictions)

	def rms(self, predictions, targets):

		return math.sqrt(sklearn.metrics.mean_squared_error(targets, predictions))

class ClassificationMetrics():

	def __new__(cls, predictions, targets):

		return cls.cm(cls, predictions, targets), cls.cr(cls, predictions, targets)

	def cm(self, predictions, targets):

		return sklearn.metrics.confusion_matrix(targets, predictions)

	def cr(self, predictions, targets):

		return sklearn.metrics.classification_report(targets, predictions)

@jit(nopython=True)
def mean(data):

	return np.mean(data)

@jit(nopython=True)
def median(data):

	return np.median(data)

@jit(nopython=True)
def stdev(data):

	return np.std(data)

@jit(nopython=True)
def variance(data):

	return np.var(data)

@jit(nopython=True)
def npmin(data):

	return np.amin(data)

@jit(nopython=True)
def npmax(data):

	return np.amax(data)

@jit(forceobj=True)
def kmeans(data, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=0.0001, precompute_distances="auto", verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm="auto"):

	kernel = sklearn.cluster.KMeans(n_clusters = n_clusters, init = init, n_init = n_init, max_iter = max_iter, tol = tol, precompute_distances = precompute_distances, verbose = verbose, random_state = random_state, copy_x = copy_x, n_jobs = n_jobs, algorithm = algorithm)
	kernel.fit(data)
	predictions = kernel.predict(data)
	centers = kernel.cluster_centers_

	return centers, predictions

@jit(forceobj=True)
def pca(data, n_components = None, copy = True, whiten = False, svd_solver = "auto", tol = 0.0, iterated_power = "auto", random_state = None):

	kernel = sklearn.decomposition.PCA(n_components = n_components, copy = copy, whiten = whiten, svd_solver = svd_solver, tol = tol, iterated_power = iterated_power, random_state = random_state)

	return kernel.fit_transform(data)

@jit(forceobj=True)
def decisiontree(data, labels, test_size = 0.3, criterion = "gini", splitter = "default", max_depth = None): #expects *2d data and 1d labels

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.tree.DecisionTreeClassifier(criterion = criterion, splitter = splitter, max_depth = max_depth)
	model = model.fit(data_train,labels_train)
	predictions = model.predict(data_test)
	metrics = ClassificationMetrics(predictions, labels_test)

	return model, metrics

class KNN:

	def knn_classifier(self, data, labels, test_size = 0.3, algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=5, p=2, weights='uniform'): #expects *2d data and 1d labels post-scaling

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.neighbors.KNeighborsClassifier()
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetrics(predictions, labels_test)

	def knn_regressor(self, data, outputs, test_size, n_neighbors = 5, weights = "uniform", algorithm = "auto", leaf_size = 30, p = 2, metric = "minkowski", metric_params = None, n_jobs = None):

		data_train, data_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(data, outputs, test_size=test_size, random_state=1)
		model = sklearn.neighbors.KNeighborsRegressor(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm, leaf_size = leaf_size, p = p, metric = metric, metric_params = metric_params, n_jobs = n_jobs)
		model.fit(data_train, outputs_train)
		predictions = model.predict(data_test)

		return model, RegressionMetrics(predictions, outputs_test)

class NaiveBayes:

	def guassian(self, data, labels, test_size = 0.3, priors = None, var_smoothing = 1e-09):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.GaussianNB(priors = priors, var_smoothing = var_smoothing)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetrics(predictions, labels_test)

	def multinomial(self, data, labels, test_size = 0.3, alpha=1.0, fit_prior=True, class_prior=None):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.MultinomialNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetrics(predictions, labels_test)

	def bernoulli(self, data, labels, test_size = 0.3, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.BernoulliNB(alpha = alpha, binarize = binarize, fit_prior = fit_prior, class_prior = class_prior)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetrics(predictions, labels_test)

	def complement(self, data, labels, test_size = 0.3, alpha=1.0, fit_prior=True, class_prior=None, norm=False):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.ComplementNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior, norm = norm)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetrics(predictions, labels_test)

class SVM:

	class CustomKernel:

		def __new__(cls, C, kernel, degre, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, random_state):

			return sklearn.svm.SVC(C = C, kernel = kernel, gamma = gamma, coef0 = coef0, shrinking = shrinking, probability = probability, tol = tol, cache_size = cache_size, class_weight = class_weight, verbose = verbose, max_iter = max_iter, decision_function_shape = decision_function_shape, random_state = random_state)

	class StandardKernel:

		def __new__(cls, kernel, C=1.0, degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):

			return sklearn.svm.SVC(C = C, kernel = kernel, gamma = gamma, coef0 = coef0, shrinking = shrinking, probability = probability, tol = tol, cache_size = cache_size, class_weight = class_weight, verbose = verbose, max_iter = max_iter, decision_function_shape = decision_function_shape, random_state = random_state)

	class PrebuiltKernel:

		class Linear:

			def __new__(cls):

				return sklearn.svm.SVC(kernel = 'linear')

		class Polynomial:

			def __new__(cls, power, r_bias):

				return sklearn.svm.SVC(kernel = 'polynomial', degree = power, coef0 = r_bias)

		class RBF:

			def __new__(cls, gamma):

				return sklearn.svm.SVC(kernel = 'rbf', gamma = gamma)

		class Sigmoid:

			def __new__(cls, r_bias):

				return sklearn.svm.SVC(kernel = 'sigmoid', coef0 = r_bias)

	def fit(self, kernel, train_data, train_outputs): # expects *2d data, 1d labels or outputs

		return kernel.fit(train_data, train_outputs)

	def eval_classification(self, kernel, test_data, test_outputs):

		predictions = kernel.predict(test_data)

		return ClassificationMetrics(predictions, test_outputs)

	def eval_regression(self, kernel, test_data, test_outputs):

		predictions = kernel.predict(test_data)

		return RegressionMetrics(predictions, test_outputs)

def random_forest_classifier(data, labels, test_size, n_estimators="warn", criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	kernel = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, bootstrap = bootstrap, oob_score = oob_score, n_jobs = n_jobs, random_state = random_state, verbose = verbose, warm_start = warm_start, class_weight = class_weight)
	kernel.fit(data_train, labels_train)
	predictions = kernel.predict(data_test)

	return kernel, ClassificationMetrics(predictions, labels_test)

def random_forest_regressor(data, outputs, test_size, n_estimators="warn", criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):

	data_train, data_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(data, outputs, test_size=test_size, random_state=1)
	kernel = sklearn.ensemble.RandomForestRegressor(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, min_impurity_split = min_impurity_split, bootstrap = bootstrap, oob_score = oob_score, n_jobs = n_jobs, random_state = random_state, verbose = verbose, warm_start = warm_start)
	kernel.fit(data_train, outputs_train)
	predictions = kernel.predict(data_test)

	return kernel, RegressionMetrics(predictions, outputs_test)

class CorrelationTests:

	def anova_oneway(self, *args): #expects arrays of samples

		results = scipy.stats.f_oneway(*args)
		return {"F-value": results[0], "p-value": results[1]}

	def pearson(self, x, y):

		results = scipy.stats.pearsonr(x, y)
		return {"r-value": results[0], "p-value": results[1]}

	def spearman(self, a, b = None, axis = 0, nan_policy = 'propagate'):

		results = scipy.stats.spearmanr(a, b = b, axis = axis, nan_policy = nan_policy)
		return {"r-value": results[0], "p-value": results[1]}

	def point_biserial(self, x,y):

		results = scipy.stats.pointbiserialr(x, y)
		return {"r-value": results[0], "p-value": results[1]}

	def kendall(self, x, y, initial_lexsort = None, nan_policy = 'propagate', method = 'auto'):

		results = scipy.stats.kendalltau(x, y, initial_lexsort = initial_lexsort, nan_policy = nan_policy, method = method)
		return {"tau": results[0], "p-value": results[1]}

	def kendall_weighted(self, x, y, rank = True, weigher = None, additive = True):

		results = scipy.stats.weightedtau(x, y, rank = rank, weigher = weigher, additive = additive)
		return {"tau": results[0], "p-value": results[1]}

	def mgc(self, x, y, compute_distance = None, reps = 1000, workers = 1, is_twosamp = False, random_state = None):

		results = scipy.stats.multiscale_graphcorr(x, y, compute_distance = compute_distance, reps = reps, workers = workers, is_twosamp = is_twosamp, random_state = random_state)
		return {"k-value": results[0], "p-value": results[1], "data": results[2]} # unsure if MGC test returns a k value

class StatisticalTests:

	def ttest_onesample(self, a, popmean, axis = 0, nan_policy = 'propagate'):

		results = scipy.stats.ttest_1samp(a, popmean, axis = axis, nan_policy = nan_policy)
		return {"t-value": results[0], "p-value": results[1]}

	def ttest_independent(self, a, b, equal = True, nan_policy = 'propagate'):

		results = scipy.stats.ttest_ind(a, b, equal_var = equal, nan_policy = nan_policy)
		return {"t-value": results[0], "p-value": results[1]}

	def ttest_statistic(self, o1, o2, equal = True):

		results = scipy.stats.ttest_ind_from_stats(o1["mean"], o1["std"], o1["nobs"], o2["mean"], o2["std"], o2["nobs"], equal_var = equal)
		return {"t-value": results[0], "p-value": results[1]}

	def ttest_related(self, a, b, axis = 0, nan_policy='propagate'):

		results = scipy.stats.ttest_rel(a, b, axis = axis, nan_policy = nan_policy)
		return {"t-value": results[0], "p-value": results[1]}

	def ks_fitness(self, rvs, cdf, args = (), N = 20, alternative = 'two-sided', mode = 'approx'):

		results = scipy.stats.kstest(rvs, cdf, args = args, N = N, alternative = alternative, mode = mode)
		return {"ks-value": results[0], "p-value": results[1]}

	def chisquare(self, f_obs, f_exp = None, ddof = None, axis = 0):

		results = scipy.stats.chisquare(f_obs, f_exp = f_exp, ddof = ddof, axis = axis)
		return {"chisquared-value": results[0], "p-value": results[1]}

	def powerdivergence(self, f_obs, f_exp = None, ddof = None, axis = 0, lambda_ = None):

		results = scipy.stats.power_divergence(f_obs, f_exp = f_exp, ddof = ddof, axis = axis, lambda_ = lambda_)
		return {"powerdivergence-value": results[0], "p-value": results[1]}

	def ks_twosample(self, x, y, alternative = 'two_sided', mode = 'auto'):
		
		results = scipy.stats.ks_2samp(x, y, alternative = alternative, mode = mode)
		return {"ks-value": results[0], "p-value": results[1]}

	def es_twosample(self, x, y, t = (0.4, 0.8)):

		results = scipy.stats.epps_singleton_2samp(x, y, t = t)
		return {"es-value": results[0], "p-value": results[1]}

	def mw_rank(self, x, y, use_continuity = True, alternative = None):

		results = scipy.stats.mannwhitneyu(x, y, use_continuity = use_continuity, alternative = alternative)
		return {"u-value": results[0], "p-value": results[1]}

	def mw_tiecorrection(self, rank_values):

		results = scipy.stats.tiecorrect(rank_values)
		return {"correction-factor": results}

	def rankdata(self, a, method = 'average'):

		results = scipy.stats.rankdata(a, method = method)
		return results

	def wilcoxon_ranksum(self, a, b): # this seems to be superceded by Mann Whitney Wilcoxon U Test

		results = scipy.stats.ranksums(a, b)
		return {"u-value": results[0], "p-value": results[1]}

	def wilcoxon_signedrank(self, x, y = None, zero_method = 'wilcox', correction = False, alternative = 'two-sided'):

		results = scipy.stats.wilcoxon(x, y = y, zero_method = zero_method, correction = correction, alternative = alternative)
		return {"t-value": results[0], "p-value": results[1]}

	def kw_htest(self, *args, nan_policy = 'propagate'):

		results = scipy.stats.kruskal(*args, nan_policy = nan_policy)
		return {"h-value": results[0], "p-value": results[1]}

	def friedman_chisquare(self, *args):

		results = scipy.stats.friedmanchisquare(*args)
		return {"chisquared-value": results[0], "p-value": results[1]}

	def bm_wtest(self, x, y, alternative = 'two-sided', distribution = 't', nan_policy = 'propagate'):

		results = scipy.stats.brunnermunzel(x, y, alternative = alternative, distribution = distribution, nan_policy = nan_policy)
		return {"w-value": results[0], "p-value": results[1]}

	def combine_pvalues(self, pvalues, method = 'fisher', weights = None):

		results = scipy.stats.combine_pvalues(pvalues, method = method, weights = weights)
		return {"combined-statistic": results[0], "p-value": results[1]}

	def jb_fitness(self, x):

		results = scipy.stats.jarque_bera(x)
		return {"jb-value": results[0], "p-value": results[1]}

	def ab_equality(self, x, y):

		results = scipy.stats.ansari(x, y)
		return {"ab-value": results[0], "p-value": results[1]}

	def bartlett_variance(self, *args):

		results = scipy.stats.bartlett(*args)
		return {"t-value": results[0], "p-value": results[1]}

	def levene_variance(self, *args, center = 'median', proportiontocut = 0.05):

		results = scipy.stats.levene(*args, center = center, proportiontocut = proportiontocut)
		return {"w-value": results[0], "p-value": results[1]}

	def sw_normality(self, x):

		results = scipy.stats.shapiro(x)
		return {"w-value": results[0], "p-value": results[1]}

	def shapiro(self, x):

		return "destroyed by facts and logic"

	def ad_onesample(self, x, dist = 'norm'):

		results = scipy.stats.anderson(x, dist = dist)
		return {"d-value": results[0], "critical-values": results[1], "significance-value": results[2]}
	
	def ad_ksample(self, samples, midrank = True):

		results = scipy.stats.anderson_ksamp(samples, midrank = midrank)
		return {"d-value": results[0], "critical-values": results[1], "significance-value": results[2]}

	def binomial(self, x, n = None, p = 0.5, alternative = 'two-sided'):

		results = scipy.stats.binom_test(x, n = n, p = p, alternative = alternative)
		return {"p-value": results}

	def fk_variance(self, *args, center = 'median', proportiontocut = 0.05):

		results = scipy.stats.fligner(*args, center = center, proportiontocut = proportiontocut)
		return {"h-value": results[0], "p-value": results[1]} # unknown if the statistic is an h value

	def mood_mediantest(self, *args, ties = 'below', correction = True, lambda_ = 1, nan_policy = 'propagate'):

		results = scipy.stats.median_test(*args, ties = ties, correction = correction, lambda_ = lambda_, nan_policy = nan_policy)
		return {"chisquared-value": results[0], "p-value": results[1], "m-value": results[2], "table": results[3]}

	def mood_equalscale(self, x, y, axis = 0):

		results = scipy.stats.mood(x, y, axis = axis)
		return {"z-score": results[0], "p-value": results[1]}

	def skewtest(self, a, axis = 0, nan_policy = 'propogate'):

		results = scipy.stats.skewtest(a, axis = axis, nan_policy = nan_policy)
		return {"z-score": results[0], "p-value": results[1]}

	def kurtosistest(self, a, axis = 0, nan_policy = 'propogate'):

		results = scipy.stats.kurtosistest(a, axis = axis, nan_policy = nan_policy)
		return {"z-score": results[0], "p-value": results[1]}

	def normaltest(self, a, axis = 0, nan_policy = 'propogate'):

		results = scipy.stats.normaltest(a, axis = axis, nan_policy = nan_policy)
		return {"z-score": results[0], "p-value": results[1]}