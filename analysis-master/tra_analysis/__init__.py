# Titan Robotics Team 2022: tra_analysis package
# Written by Arthur Lu, Jacob Levine, Dev Singh, and James Pan
# Notes:
#    this should be imported as a python package using 'import tra_analysis'
#    this should be included in the local directory or environment variable
#    this module has been optimized for multhreaded computing
#    current benchmark of optimization: 1.33 times faster
# setup:

__version__ = "3.0.0"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
	3.0.0:
		- incremented version to release 3.0.0
	3.0.0-rc2:
		- fixed __changelog__
		- fixed __all__ of Analysis, Array, ClassificationMetric, CorrelationTest, RandomForest, Sort, SVM
		- populated __all__
	3.0.0-alpha.4:
		- changed version to 3 because of significant changes
		- added backwards compatibility import of analysis
	2.1.0-alpha.3:
		- fixed indentation in meta data
	2.1.0-alpha.2:
		- updated SVM import
	2.1.0-alpha.1:
		- moved multiple submodules under analysis to their own modules/files
		- added header, __version__, __changelog__, __author__, __all__ (unpopulated)
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
	"Jacob Levine <jlevine@imsa.edu>",
	"Dev Singh <dev@devksingh.com>",
	"James Pan <zpan@imsa.edu>"
)

__all__ = [
	"Analysis",
	"Array",
	"ClassificationMetric",
	"CorrelationTest",
	"Expression",
	"Fit",
	"KNN",
	"NaiveBayes",
	"RandomForest",
	"RegressionMetric",
	"Sort",
	"StatisticalTest",
	"SVM"
]

from . import Analysis as Analysis
from . import Analysis as analysis
from .Array import Array
from .ClassificationMetric import ClassificationMetric
from . import CorrelationTest
from .equation import Expression
from . import Fit
from . import KNN
from . import NaiveBayes
from . import RandomForest
from .RegressionMetric import RegressionMetric
from . import Sort
from . import StatisticalTest
from . import SVM