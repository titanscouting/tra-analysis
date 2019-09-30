# Titan Robotics Team 2022: Data Analysis Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#   this should be imported as a python module using 'import analysis'
#   this should be included in the local directory or environment variable
#   this module has not been optimized for multhreaded computing
#   current benchmark of optimization: 1.33 times faster
# setup:

__version__ = "1.1.1.001"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
1.1.1.001:
    - regression_engine() bug fixes, now actaully regresses
1.1.1.000:
    - added regression_engine()
    - added all regressions except polynomial
1.1.0.007:
    - updated _init_device()
1.1.0.006:
    - removed useless try statements
1.1.0.005:
    - removed impossible outcomes
1.1.0.004:
    - added performance metrics (r^2, mse, rms)
1.1.0.003:
    - resolved nopython mode for mean, median, stdev, variance
1.1.0.002:
    - snapped (removed) majority of uneeded imports
    - forced object mode (bad) on all jit
    - TODO: stop numba complaining about not being able to compile in nopython mode
1.1.0.001:
    - removed from sklearn import * to resolve uneeded wildcard imports
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
    "Arthur Lu <arthurlu@ttic.edu>",
    "Jacob Levine <jlevine@ttic.edu>",
)

__all__ = [
    '_init_device',
    'load_csv',
    'basic_stats',
    'z_score',
    'z_normalize',
    'histo_analysis',
    'r_squared',
    'mse',
    'rms',
    # all statistics functions left out due to integration in other functions
]

# now back to your regularly scheduled programming:

# imports (now in alphabetical order! v 1.0.3.006):

import csv
import numba
from numba import jit
import numpy as np
import math
from analysis import regression
from sklearn import metrics
from sklearn import preprocessing
import torch

class error(ValueError):
    pass

def _init_device():  # initiates computation device for ANNs
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

@jit(forceobj=True)
def load_csv(filepath):
    with open(filepath, newline='') as csvfile:
        file_array = np.array(list(csv.reader(csvfile)))
        csvfile.close()
    return file_array

# expects 1d array
@jit(forceobj=True)
def basic_stats(data):

    data_t = np.array(data).astype(float)

    _mean = mean(data_t)
    _median = median(data_t)
    _stdev = stdev(data_t)
    _variance = variance(data_t)

    return _mean, _median, _stdev, _variance

# returns z score with inputs of point, mean and standard deviation of spread
@jit(forceobj=True)
def z_score(point, mean, stdev):
    score = (point - mean) / stdev
    return score

# expects 2d array, normalizes across all axes
@jit(forceobj=True)
def z_normalize(array, *args):

   array = np.array(array)

   for arg in args:

       array = preprocessing.normalize(array, axis = arg)

   return array

@jit(forceobj=True)
# expects 2d array of [x,y]
def histo_analysis(hist_data):

    hist_data = np.array(hist_data)

    derivative = np.array(len(hist_data) - 1, dtype = float)

    t = np.diff(hist_data)

    derivative = t[1] / t[0]

    np.sort(derivative)

    return basic_stats(derivative)[0], basic_stats(derivative)[3]

@jit(forceobj=True)
def regression_engine(device, inputs, outputs, args, loss = torch.nn.MSELoss(), _iterations = 10000, lr = 0.01):

    regressions = []

    if 'cuda' in device:

        regression.set_device(device)

        if 'linear' in args:

            model = regression.SGDTrain(regression.LinearRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor([outputs]).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'log' in args:

            model = regression.SGDTrain(regression.LogRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor(outputs).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'exp' in args:

            model = regression.SGDTrain(regression.ExpRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor(outputs).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        #if 'poly' in args:

            #TODO because Jacob hasnt fixed regression.py

        if 'sig' in args:

            model = regression.SGDTrain(regression.SigmoidalRegKernelArthur(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor(outputs).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

    else:

        regression.set_device(device)

        if 'linear' in args:

            model = regression.SGDTrain(regression.LinearRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'log' in args:

            model = regression.SGDTrain(regression.LogRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'exp' in args:

            model = regression.SGDTrain(regression.ExpRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        #if 'poly' in args:

            #TODO because Jacob hasnt fixed regression.py

        if 'sig' in args:

            model = regression.SGDTrain(regression.SigmoidalRegKernelArthur(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

    return regressions

@jit(forceobj=True)
def r_squared(predictions, targets):  # assumes equal size inputs

    return metrics.r2_score(np.array(targets), np.array(predictions))

@jit(forceobj=True)
def mse(predictions, targets):

    return metrics.mean_squared_error(np.array(targets), np.array(predictions))

@jit(forceobj=True)
def rms(predictions, targets):

    return math.sqrt(metrics.mean_squared_error(np.array(targets), np.array(predictions)))

@jit(nopython=True)
def mean(data):

    return np.mean(data)

@jit(nopython=True)
def median(data):

    return np.median(data)

@jit(nopython=True)
def stdev(data):

    return np.std(data)

@jit(nopython=True)
def variance(data):

    return np.var(data)