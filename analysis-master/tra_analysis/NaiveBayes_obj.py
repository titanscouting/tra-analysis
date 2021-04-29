# Only included for backwards compatibility! Do not update, NaiveBayes is preferred and supported.

import sklearn
from sklearn import model_selection, naive_bayes
from . import ClassificationMetric, RegressionMetric

class NaiveBayes:

	def guassian(self, data, labels, test_size = 0.3, priors = None, var_smoothing = 1e-09):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.GaussianNB(priors = priors, var_smoothing = var_smoothing)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetric(predictions, labels_test)

	def multinomial(self, data, labels, test_size = 0.3, alpha=1.0, fit_prior=True, class_prior=None):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.MultinomialNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetric(predictions, labels_test)

	def bernoulli(self, data, labels, test_size = 0.3, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.BernoulliNB(alpha = alpha, binarize = binarize, fit_prior = fit_prior, class_prior = class_prior)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetric(predictions, labels_test)

	def complement(self, data, labels, test_size = 0.3, alpha=1.0, fit_prior=True, class_prior=None, norm=False):

		data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=test_size, random_state=1)
		model = sklearn.naive_bayes.ComplementNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior, norm = norm)
		model.fit(data_train, labels_train)
		predictions = model.predict(data_test)

		return model, ClassificationMetric(predictions, labels_test)