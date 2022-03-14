# Titan Robotics Team 2022: ClassificationMetric submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import ClassificationMetric'
# setup:

__version__ = "1.0.2"

__changelog__ = """changelog:
	1.0.2:
		- optimized imports
	1.0.1:
		- fixed __all__
	1.0.0:
		- ported analysis.ClassificationMetric() here
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	"ClassificationMetric",
]

import sklearn

class ClassificationMetric():

	def __new__(cls, predictions, targets):

		return cls.cm(cls, predictions, targets), cls.cr(cls, predictions, targets)

	def cm(self, predictions, targets):

		return sklearn.metrics.confusion_matrix(targets, predictions)

	def cr(self, predictions, targets):

		return sklearn.metrics.classification_report(targets, predictions)