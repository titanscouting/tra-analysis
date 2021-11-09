# Titan Robotics Team 2022: SVM submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import SVM'
# setup:

__version__ = "1.0.3"

__changelog__ = """changelog:
	1.0.3:
		- optimized imports
	1.0.2: 
		- fixed __all__
	1.0.1:
		- removed unessasary self calls
		- removed classness
	1.0.0:
		- ported analysis.SVM() here
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	"CustomKernel",
	"StandardKernel",
	"PrebuiltKernel",
	"fit",
	"eval_classification",
	"eval_regression",
]

import sklearn
from . import ClassificationMetric, RegressionMetric

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

def fit(kernel, train_data, train_outputs): # expects *2d data, 1d labels or outputs

	return kernel.fit(train_data, train_outputs)

def eval_classification(kernel, test_data, test_outputs):

	predictions = kernel.predict(test_data)

	return ClassificationMetric(predictions, test_outputs)

def eval_regression(kernel, test_data, test_outputs):

	predictions = kernel.predict(test_data)

	return RegressionMetric(predictions, test_outputs)