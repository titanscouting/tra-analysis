# Titan Robotics Team 2022: tra_analysis package
# Written by Arthur Lu, Jacob Levine, and Dev Singh
# Notes:
#    this should be imported as a python package using 'import tra_analysis'
#    this should be included in the local directory or environment variable
#    this module has been optimized for multhreaded computing
#    current benchmark of optimization: 1.33 times faster
# setup:

__version__ = "2.1.0-alpha.1"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
    2.1.0-alpha.1:
        - moved multiple submodules under analysis to their own modules/files
        - added header, __version__, __changelog__, __author__, __all__ (unpopulated)
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
    "Jacob Levine <jlevine@imsa.edu>",
    "Dev Singh <dev@devksingh.com>",
)

__all__ = [
]

from . import Analysis
from .Array import Array
from .ClassificationMetric import ClassificationMetric
from . import CorrelationTest
from . import Fit
from . import KNN
from . import NaiveBayes
from . import RandomForest
from .RegressionMetric import RegressionMetric
from . import Sort
from . import StatisticalTest
from .SVM import SVM