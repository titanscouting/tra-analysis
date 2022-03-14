# Titan Robotics Team 2022: RegressionMetric submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import RegressionMetric'
# setup:

__version__ = "1.0.1"

__changelog__ = """changelog:
	1.0.1:
		- optimized imports
	1.0.0:
		- ported analysis.RegressionMetric() here
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	'RegressionMetric'
]

import numpy as np
import sklearn

class RegressionMetric():

	def __new__(cls, predictions, targets):

		return cls.r_squared(cls, predictions, targets), cls.mse(cls, predictions, targets), cls.rms(cls, predictions, targets)

	def r_squared(self, predictions, targets):  # assumes equal size inputs

		return sklearn.metrics.r2_score(targets, predictions)

	def mse(self, predictions, targets):

		return sklearn.metrics.mean_squared_error(targets, predictions)

	def rms(self, predictions, targets):

		return np.sqrt(sklearn.metrics.mean_squared_error(targets, predictions))