# Titan Robotics Team 2022: RandomForest submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import RandomForest'
# setup:

__version__ = "1.0.3"

__changelog__ = """changelog:
	1.0.3:
		- updated RandomForestClassifier and RandomForestRegressor parameters to match sklearn v 1.0.2
		- changed default values for kwargs to rely on sklearn
	1.0.2:
		- optimized imports
	1.0.1:
		- fixed __all__
	1.0.0:
		- ported analysis.RandomFores() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	"random_forest_classifier",
	"random_forest_regressor",
]

import sklearn, sklearn.ensemble, sklearn.naive_bayes
from . import ClassificationMetric, RegressionMetric

def random_forest_classifier(data, labels, test_size, n_estimators, **kwargs):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	kernel = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, **kwargs)
	kernel.fit(data_train, labels_train)
	predictions = kernel.predict(data_test)

	return kernel, ClassificationMetric(predictions, labels_test)

def random_forest_regressor(data, outputs, test_size, n_estimators, **kwargs):

	data_train, data_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(data, outputs, test_size=test_size, random_state=1)
	kernel = sklearn.ensemble.RandomForestRegressor(n_estimators = n_estimators, **kwargs)
	kernel.fit(data_train, outputs_train)
	predictions = kernel.predict(data_test)

	return kernel, RegressionMetric.RegressionMetric(predictions, outputs_test)