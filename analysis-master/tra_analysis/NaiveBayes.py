# Titan Robotics Team 2022: NaiveBayes submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import NaiveBayes'
# setup:

__version__ = "1.0.0"

__changelog__ = """changelog:
	1.0.0:
		- ported analysis.NaiveBayes() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	'gaussian',
	'multinomial'
	'bernoulli',
	'complement'
]

import sklearn
from sklearn import model_selection, naive_bayes
from . import ClassificationMetric, RegressionMetric

def gaussian(data, labels, test_size = 0.3, priors = None, var_smoothing = 1e-09):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.GaussianNB(priors = priors, var_smoothing = var_smoothing)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)

def multinomial(data, labels, test_size = 0.3, alpha=1.0, fit_prior=True, class_prior=None):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.MultinomialNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)

def bernoulli(data, labels, test_size = 0.3, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.BernoulliNB(alpha = alpha, binarize = binarize, fit_prior = fit_prior, class_prior = class_prior)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)

def complement(data, labels, test_size = 0.3, alpha=1.0, fit_prior=True, class_prior=None, norm=False):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.ComplementNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior, norm = norm)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)