# Only included for backwards compatibility! Do not update, NaiveBayes is preferred and supported.

import sklearn
from sklearn import model_selection, neighbors
from . import ClassificationMetric, RegressionMetric

class KNN:

	def knn_classifier(self, data, labels, n_neighbors, test_size = 0.3, algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, p=2, weights='uniform'): #expects *2d data and 1d labels post-scaling

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.neighbors.KNeighborsClassifier()
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetric(predictions, labels_test)

	def knn_regressor(self, data, outputs, n_neighbors, test_size = 0.3, weights = "uniform", algorithm = "auto", leaf_size = 30, p = 2, metric = "minkowski", metric_params = None, n_jobs = None):

		data_train, data_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(data, outputs, test_size=test_size, random_state=1)
		model = sklearn.neighbors.KNeighborsRegressor(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm, leaf_size = leaf_size, p = p, metric = metric, metric_params = metric_params, n_jobs = n_jobs)
		model.fit(data_train, outputs_train)
		predictions = model.predict(data_test)

		return model, RegressionMetric(predictions, outputs_test)