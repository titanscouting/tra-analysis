# Titan Robotics Team 2022: RandomForest submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import RandomForest'
# setup:

__version__ = "1.0.1"

__changelog__ = """changelog:
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

import sklearn
from sklearn import ensemble, model_selection
from . import ClassificationMetric, RegressionMetric

def random_forest_classifier(data, labels, test_size, n_estimators, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None):

	data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
	kernel = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, bootstrap = bootstrap, oob_score = oob_score, n_jobs = n_jobs, random_state = random_state, verbose = verbose, warm_start = warm_start, class_weight = class_weight)
	kernel.fit(data_train, labels_train)
	predictions = kernel.predict(data_test)

	return kernel, ClassificationMetric(predictions, labels_test)

def random_forest_regressor(data, outputs, test_size, n_estimators, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):

	data_train, data_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(data, outputs, test_size=test_size, random_state=1)
	kernel = sklearn.ensemble.RandomForestRegressor(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, min_impurity_split = min_impurity_split, bootstrap = bootstrap, oob_score = oob_score, n_jobs = n_jobs, random_state = random_state, verbose = verbose, warm_start = warm_start)
	kernel.fit(data_train, outputs_train)
	predictions = kernel.predict(data_test)

	return kernel, RegressionMetric.RegressionMetric(predictions, outputs_test)