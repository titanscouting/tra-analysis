# Titan Robotics Team 2022: KNN submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import KNN'
# setup:

__version__ = "1.0.2"

__changelog__ = """changelog:
	1.0.2:
		- generalized optional args to **kwargs
	1.0.1:
		- optimized imports
	1.0.0:
		- ported analysis.KNN() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
	"James Pan <zpan@imsa.edu>"
)

__all__ = [
	'knn_classifier',
	'knn_regressor'
]

import sklearn
from . import ClassificationMetric, RegressionMetric

def knn_classifier(data, labels, n_neighbors = 5, test_size = 0.3, **kwargs): #expects *2d data and 1d labels post-scaling

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, **kwargs)
	model.fit(data_train, labels_train)
	predictions = model.predict(data_test)

	return model, ClassificationMetric(predictions, labels_test)

def knn_regressor(data, outputs, n_neighbors = 5, test_size = 0.3, **kwargs):

	data_train, data_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(data, outputs, test_size=test_size, random_state=1)
	model = sklearn.neighbors.KNeighborsRegressor(n_neighbors = n_neighbors, **kwargs)
	model.fit(data_train, outputs_train)
	predictions = model.predict(data_test)

	return model, RegressionMetric.RegressionMetric(predictions, outputs_test)