# Titan Robotics Team 2022: NaiveBayes submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import NaiveBayes'
# setup:

__version__ = "1.0.2"

__changelog__ = """changelog:
	1.0.2:
		- generalized optional args to **kwargs
	1.0.1:
		- optimized imports
	1.0.0:
		- ported analysis.NaiveBayes() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	'gaussian',
	'multinomial',
	'bernoulli',
	'complement',
]

import sklearn
from . import ClassificationMetric

def gaussian(data, labels, test_size = 0.3, **kwargs):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.GaussianNB(**kwargs)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)

def multinomial(data, labels, test_size = 0.3, **kwargs):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.MultinomialNB(**kwargs)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)

def bernoulli(data, labels, test_size = 0.3, **kwargs):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.BernoulliNB(**kwargs)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)

def complement(data, labels, test_size = 0.3, **kwargs):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.naive_bayes.ComplementNB(**kwargs)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)