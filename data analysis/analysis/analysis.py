# Titan Robotics Team 2022: Data Analysis Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#   this should be imported as a python module using 'import analysis'
#   this should be included in the local directory or environment variable
#   this module has not been optimized for multhreaded computing
# number of easter eggs: 2
# setup:

__version__ = "1.1.0.000"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
1.1.0.000:
    - removed c_entities,nc_entities,obstacles,objectives from __all__
    - applied numba.jit to all functions
    - depreciated and removed stdev_z_split
    - cleaned up histo_analysis to include numpy and numba.jit optimizations
    - depreciated and removed all regression functions in favor of future pytorch optimizer
    - depreciated and removed all nonessential functions (basic_analysis, benchmark, strip_data)
    - optimized z_normalize using sklearn.preprocessing.normalize
    - TODO: implement kernel/function based pytorch regression optimizer
1.0.9.000:
    - refactored
    - numpyed everything
    - removed stats in favor of numpy functions
1.0.8.005:
    - minor fixes
1.0.8.004:
    - removed a few unused dependencies
1.0.8.003:
    - added p_value function
1.0.8.002:
    - updated __all__ correctly to contain changes made in v 1.0.8.000 and v 1.0.8.001
1.0.8.001:
    - refactors
    - bugfixes
1.0.8.000:
    - depreciated histo_analysis_old
    - depreciated debug
    - altered basic_analysis to take array data instead of filepath
    - refactor
    - optimization
1.0.7.002:
    - bug fixes
1.0.7.001:
    - bug fixes
1.0.7.000:
    - added tanh_regression (logistical regression)
    - bug fixes
1.0.6.005:
    - added z_normalize function to normalize dataset
    - bug fixes
1.0.6.004:
    - bug fixes
1.0.6.003:
    - bug fixes
1.0.6.002:
    - bug fixes
1.0.6.001:
    - corrected __all__ to contain all of the functions
1.0.6.000:
    - added calc_overfit, which calculates two measures of overfit, error and performance
    - added calculating overfit to optimize_regression
1.0.5.000:
    - added optimize_regression function, which is a sample function to find the optimal regressions
    - optimize_regression function filters out some overfit funtions (functions with r^2 = 1)
    - planned addition: overfit detection in the optimize_regression function
1.0.4.002:
    - added __changelog__
    - updated debug function with log and exponential regressions
1.0.4.001:
    - added log regressions
    - added exponential regressions
    - added log_regression and exp_regression to __all__
1.0.3.008:
    - added debug function to further consolidate functions
1.0.3.007:
    - added builtin benchmark function
    - added builtin random (linear) data generation function
    - added device initialization (_init_device)
1.0.3.006:
    - reorganized the imports list to be in alphabetical order
    - added search and regurgitate functions to c_entities, nc_entities, obstacles, objectives
1.0.3.005:
    - major bug fixes
    - updated historical analysis
    - depreciated old historical analysis
1.0.3.004:
    - added __version__, __author__, __all__
    - added polynomial regression
    - added root mean squared function
    - added r squared function
1.0.3.003:
    - bug fixes
    - added c_entities
1.0.3.002:
    - bug fixes
    - added nc_entities, obstacles, objectives
    - consolidated statistics.py to analysis.py
1.0.3.001:
    - compiled 1d, column, and row basic stats into basic stats function
1.0.3.000:
    - added historical analysis function
1.0.2.xxx:
    - added z score test
1.0.1.xxx:
    - major bug fixes
1.0.0.xxx:
    - added loading csv
    - added 1d, column, row basic stats
"""

__author__ = (
    "Arthur Lu <arthurlu@ttic.edu>, "
    "Jacob Levine <jlevine@ttic.edu>,"
)

__all__ = [
    '_init_device',
    'load_csv',
    'basic_stats',
    'z_score',
    'z_normalize',
    'histo_analysis',
    # all statistics functions left out due to integration in other functions
]

# now back to your regularly scheduled programming:

# imports (now in alphabetical order! v 1.0.3.006):

from bisect import bisect_left, bisect_right
import collections
import csv
from decimal import Decimal
import functools
from fractions import Fraction
from itertools import groupby
import math
import matplotlib
import numba
from numba import jit
import numbers
import numpy as np
import pandas
import random
import scipy
from scipy.optimize import curve_fit
from scipy import stats
from sklearn import preprocessing
from sklearn import *
# import statistics <-- statistics.py functions have been integrated into analysis.py as of v 1.0.3.002
import time
import torch

class error(ValueError):
    pass

def _init_device(setting, arg):  # initiates computation device for ANNs
    if setting == "cuda":
        try:
            return torch.device(setting + ":" + str(arg) if torch.cuda.is_available() else "cpu")
        except:
            raise error("could not assign cuda or cpu")
    elif setting == "cpu":
        try:
            return torch.device("cpu")
        except:
            raise error("could not assign cpu")
    else:
        raise error("specified device does not exist")

@jit
def load_csv(filepath):
    with open(filepath, newline='') as csvfile:
        file_array = np.array(list(csv.reader(csvfile)))
        csvfile.close()
    return file_array

# data=array, mode = ['1d':1d_basic_stats, 'column':c_basic_stats, 'row':r_basic_stats], arg for mode 1 or mode 2 for column or row
@jit
def basic_stats(data):

    data_t = np.array(data).astype(float)

    _mean = mean(data_t)
    _median = median(data_t)
    _stdev = stdev(data_t)
    _variance = variance(data_t)

    return _mean, _median, _stdev, _variance

# returns z score with inputs of point, mean and standard deviation of spread
@jit
def z_score(point, mean, stdev):
    score = (point - mean) / stdev
    return score

# expects 2d array, normalizes across all axes
@jit
def z_normalize(array, *args):

   array = np.array(array)

   for arg in args:

       array = preprocessing.normalize(array, axis = arg)

   return array

@jit
# expects 2d array of [x,y]
def histo_analysis(hist_data):

    hist_data = np.array(hist_data)

    derivative = np.array(len(hist_data) - 1, dtype = float)

    t = np.diff(hist_data)

    derivative = t[1] / t[0]

    np.sort(derivative)
    mean_derivative = basic_stats(derivative)[0]
    stdev_derivative = basic_stats(derivative)[3]

    return mean_derivative, stdev_derivative

@jit
def mean(data):

    return np.mean(data)

@jit
def median(data):

    return np.median(data)

@jit
def stdev(data):

    return np.std(data)

@jit
def variance(data):

    return np.var(data)