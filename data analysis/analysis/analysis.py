# Titan Robotics Team 2022: Data Analysis Module
# Written by Arthur Lu & Jacob Levine
# Notes:
#   this should be imported as a python module using 'import analysis'
#   this should be included in the local directory or environment variable
#   this module has been optimized for multhreaded computing
#   current benchmark of optimization: 1.33 times faster
# setup:

__version__ = "1.1.3.002"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
1.1.3.002:
    - renamed regression class to Regression, regression_engine() to regression gliko2_engine class to Gliko2
1.1.3.001:
    - changed glicko2() to return tuple instead of array
1.1.3.000:
    - added glicko2_engine class and glicko()
    - verified glicko2() accuracy
1.1.2.003:
    - fixed elo()
1.1.2.002:
    - added elo()
    - elo() has bugs to be fixed
1.1.2.001:
    - readded regrression import
1.1.2.000:
    - integrated regression.py as regression class
    - removed regression import
    - fixed metadata for regression class
    - fixed metadata for analysis class
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
    "Arthur Lu <learthurgo@gmail.com>",
    "Jacob Levine <jlevine@imsa.edu>",
)

__all__ = [
    '_init_device',
    'load_csv',
    'basic_stats',
    'z_score',
    'z_normalize',
    'histo_analysis',
    'regression',
    'elo',
    'gliko2',
    'r_squared',
    'mse',
    'rms',
    'Regression',
    'Gliko2'
    # all statistics functions left out due to integration in other functions
]

# now back to your regularly scheduled programming:

# imports (now in alphabetical order! v 1.0.3.006):

import csv
import numba
from numba import jit
import numpy as np
import math
try:
    from analysis import regression
except:
    pass
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
def regression(device, inputs, outputs, args, loss = torch.nn.MSELoss(), _iterations = 10000, lr = 0.01): # inputs, outputs expects N-D array 

    regressions = []

    if 'cuda' in device:

        Regression.set_device(device)

        if 'linear' in args:

            model = Regression.SGDTrain(Regression.LinearRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor([outputs]).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'log' in args:

            model = Regression.SGDTrain(Regression.LogRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor(outputs).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'exp' in args:

            model = Regression.SGDTrain(Regression.ExpRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor(outputs).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        #if 'poly' in args:

            #TODO because Jacob hasnt fixed regression.py

        if 'sig' in args:

            model = Regression.SGDTrain(Regression.SigmoidalRegKernelArthur(len(inputs)), torch.tensor(inputs).to(torch.float).cuda(), torch.tensor(outputs).to(torch.float).cuda(), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

    else:

        Regression.set_device(device)

        if 'linear' in args:

            model = Regression.SGDTrain(Regression.LinearRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'log' in args:

            model = Regression.SGDTrain(Regression.LogRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        if 'exp' in args:

            model = Regression.SGDTrain(Regression.ExpRegKernel(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

        #if 'poly' in args:

            #TODO because Jacob hasnt fixed regression.py

        if 'sig' in args:

            model = Regression.SGDTrain(Regression.SigmoidalRegKernelArthur(len(inputs)), torch.tensor(inputs).to(torch.float), torch.tensor(outputs).to(torch.float), iterations=_iterations, learning_rate=lr, return_losses=True)
            regressions.append([model[0].parameters, model[1][::-1][0]])

    return regressions

@jit(nopython=True)
def elo(starting_score, opposing_scores, observed, N, K):

    expected = 1/(1+10**((np.array(opposing_scores) - starting_score)/N))

    return starting_score + K*(np.sum(observed) - np.sum(expected))

@jit(forceobj=True)
def gliko2(starting_score, starting_rd, starting_vol, opposing_scores, opposing_rd, observations):

    player = Gliko2(rating = starting_score, rd = starting_rd, vol = starting_vol)

    player.update_player([x for x in opposing_scores], [x for x in opposing_rd], observations)

    return (player.rating, player.rd, player.vol)

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

class Regression:

    # Titan Robotics Team 2022: CUDA-based Regressions Module
    # Written by Arthur Lu & Jacob Levine
    # Notes:
    #   this module has been automatically inegrated into analysis.py, and should be callable as a class from the package
    #   this module is cuda-optimized and vectorized (except for one small part)
    # setup:

    __version__ = "1.0.0.002"

    # changelog should be viewed using print(analysis.regression.__changelog__)
    __changelog__ = """
    1.0.0.002:
        -Added more parameters to log, exponential, polynomial
        -Added SigmoidalRegKernelArthur, because Arthur apparently needs
        to train the scaling and shifting of sigmoids

    1.0.0.001:
        -initial release, with linear, log, exponential, polynomial, and sigmoid kernels
        -already vectorized (except for polynomial generation) and CUDA-optimized
    """

    __author__ = (
        "Jacob Levine <jlevine@imsa.edu>",
        "Arthur Lu <learthurgo@gmail.com>"
    )

    __all__ = [
        'factorial',
        'take_all_pwrs',
        'num_poly_terms',
        'set_device',
        'LinearRegKernel',
        'SigmoidalRegKernel',
        'LogRegKernel',
        'PolyRegKernel',
        'ExpRegKernel',
        'SigmoidalRegKernelArthur',
        'SGDTrain',
        'CustomTrain'
    ]


    # imports (just one for now):

    import torch

    device = "cuda:0" if torch.torch.cuda.is_available() else "cpu"

    #todo: document completely

    def factorial(n):
        if n==0:
            return 1
        else:
            return n*factorial(n-1)
    def num_poly_terms(num_vars, power):
        if power == 0:
            return 0
        return int(factorial(num_vars+power-1) / factorial(power) / factorial(num_vars-1)) + num_poly_terms(num_vars, power-1)

    def take_all_pwrs(vec,pwr):
        #todo: vectorize (kinda)
        combins=torch.combinations(vec, r=pwr, with_replacement=True)
        out=torch.ones(combins.size()[0])
        for i in torch.t(combins):
            out *= i
        return torch.cat(out,take_all_pwrs(vec, pwr-1))

    def set_device(new_device):
        global device
        device=new_device

    class LinearRegKernel():
        parameters= []
        weights=None
        bias=None
        def __init__(self, num_vars):
            self.weights=torch.rand(num_vars, requires_grad=True, device=device)
            self.bias=torch.rand(1, requires_grad=True, device=device)
            self.parameters=[self.weights,self.bias]
        def forward(self,mtx):
            long_bias=self.bias.repeat([1,mtx.size()[1]])
            return torch.matmul(self.weights,mtx)+long_bias

    class SigmoidalRegKernel():
        parameters= []
        weights=None
        bias=None
        sigmoid=torch.nn.Sigmoid()
        def __init__(self, num_vars):
            self.weights=torch.rand(num_vars, requires_grad=True, device=device)
            self.bias=torch.rand(1, requires_grad=True, device=device)
            self.parameters=[self.weights,self.bias]
        def forward(self,mtx):
            long_bias=self.bias.repeat([1,mtx.size()[1]])
            return self.sigmoid(torch.matmul(self.weights,mtx)+long_bias)

    class SigmoidalRegKernelArthur():
        parameters= []
        weights=None
        in_bias=None
        scal_mult=None
        out_bias=None
        sigmoid=torch.nn.Sigmoid()
        def __init__(self, num_vars):
            self.weights=torch.rand(num_vars, requires_grad=True, device=device)
            self.in_bias=torch.rand(1, requires_grad=True, device=device)
            self.scal_mult=torch.rand(1, requires_grad=True, device=device)
            self.out_bias=torch.rand(1, requires_grad=True, device=device)
            self.parameters=[self.weights,self.in_bias, self.scal_mult, self.out_bias]
        def forward(self,mtx):
            long_in_bias=self.in_bias.repeat([1,mtx.size()[1]])
            long_out_bias=self.out_bias.repeat([1,mtx.size()[1]])
            return (self.scal_mult*self.sigmoid(torch.matmul(self.weights,mtx)+long_in_bias))+long_out_bias

    class LogRegKernel():
        parameters= []
        weights=None
        in_bias=None
        scal_mult=None
        out_bias=None
        def __init__(self, num_vars):
            self.weights=torch.rand(num_vars, requires_grad=True, device=device)
            self.in_bias=torch.rand(1, requires_grad=True, device=device)
            self.scal_mult=torch.rand(1, requires_grad=True, device=device)
            self.out_bias=torch.rand(1, requires_grad=True, device=device)
            self.parameters=[self.weights,self.in_bias, self.scal_mult, self.out_bias]
        def forward(self,mtx):
            long_in_bias=self.in_bias.repeat([1,mtx.size()[1]])
            long_out_bias=self.out_bias.repeat([1,mtx.size()[1]])
            return (self.scal_mult*torch.log(torch.matmul(self.weights,mtx)+long_in_bias))+long_out_bias

    class ExpRegKernel():
        parameters= []
        weights=None
        in_bias=None
        scal_mult=None
        out_bias=None
        def __init__(self, num_vars):
            self.weights=torch.rand(num_vars, requires_grad=True, device=device)
            self.in_bias=torch.rand(1, requires_grad=True, device=device)
            self.scal_mult=torch.rand(1, requires_grad=True, device=device)
            self.out_bias=torch.rand(1, requires_grad=True, device=device)
            self.parameters=[self.weights,self.in_bias, self.scal_mult, self.out_bias]
        def forward(self,mtx):
            long_in_bias=self.in_bias.repeat([1,mtx.size()[1]])
            long_out_bias=self.out_bias.repeat([1,mtx.size()[1]])
            return (self.scal_mult*torch.exp(torch.matmul(self.weights,mtx)+long_in_bias))+long_out_bias

    class PolyRegKernel():
        parameters= []
        weights=None
        bias=None
        power=None
        def __init__(self, num_vars, power):
            self.power=power
            num_terms=num_poly_terms(num_vars, power)
            self.weights=torch.rand(num_terms, requires_grad=True, device=device)
            self.bias=torch.rand(1, requires_grad=True, device=device)
            self.parameters=[self.weights,self.bias]
        def forward(self,mtx):
            #TODO: Vectorize the last part
            cols=[]
            for i in torch.t(mtx):
                cols.append(take_all_pwrs(i,self.power))
            new_mtx=torch.t(torch.stack(cols))
            long_bias=self.bias.repeat([1,mtx.size()[1]])
            return torch.matmul(self.weights,new_mtx)+long_bias

    def SGDTrain(kernel, data, ground, loss=torch.nn.MSELoss(), iterations=1000, learning_rate=.1, return_losses=False):
        optim=torch.optim.SGD(kernel.parameters, lr=learning_rate)
        data_cuda=data.to(device)
        ground_cuda=ground.to(device)
        if (return_losses):
            losses=[]
            for i in range(iterations):
                with torch.set_grad_enabled(True):
                    optim.zero_grad()
                    pred=kernel.forward(data_cuda)
                    ls=loss(pred,ground_cuda)
                    losses.append(ls.item())
                    ls.backward()
                    optim.step()
            return [kernel,losses]
        else:
            for i in range(iterations):
                with torch.set_grad_enabled(True):
                    optim.zero_grad()
                    pred=kernel.forward(data_cuda)
                    ls=loss(pred,ground_cuda)
                    ls.backward()
                    optim.step()
            return kernel

    def CustomTrain(kernel, optim, data, ground, loss=torch.nn.MSELoss(), iterations=1000, return_losses=False):
        data_cuda=data.to(device)
        ground_cuda=ground.to(device)
        if (return_losses):
            losses=[]
            for i in range(iterations):
                with torch.set_grad_enabled(True):
                    optim.zero_grad()
                    pred=kernel.forward(data)
                    ls=loss(pred,ground)
                    losses.append(ls.item())
                    ls.backward()
                    optim.step()
            return [kernel,losses]
        else:
            for i in range(iterations):
                with torch.set_grad_enabled(True):
                    optim.zero_grad()
                    pred=kernel.forward(data_cuda)
                    ls=loss(pred,ground_cuda)
                    ls.backward()
                    optim.step()
            return kernel

class Gliko2:

    _tau = 0.5

    def getRating(self):
        return (self.__rating * 173.7178) + 1500 

    def setRating(self, rating):
        self.__rating = (rating - 1500) / 173.7178

    rating = property(getRating, setRating)

    def getRd(self):
        return self.__rd * 173.7178

    def setRd(self, rd):
        self.__rd = rd / 173.7178

    rd = property(getRd, setRd)
     
    def __init__(self, rating = 1500, rd = 350, vol = 0.06):

        self.setRating(rating)
        self.setRd(rd)
        self.vol = vol
            
    def _preRatingRD(self):

        self.__rd = math.sqrt(math.pow(self.__rd, 2) + math.pow(self.vol, 2))
        
    def update_player(self, rating_list, RD_list, outcome_list):

        rating_list = [(x - 1500) / 173.7178 for x in rating_list]
        RD_list = [x / 173.7178 for x in RD_list]

        v = self._v(rating_list, RD_list)
        self.vol = self._newVol(rating_list, RD_list, outcome_list, v)
        self._preRatingRD()
        
        self.__rd = 1 / math.sqrt((1 / math.pow(self.__rd, 2)) + (1 / v))
        
        tempSum = 0
        for i in range(len(rating_list)):
            tempSum += self._g(RD_list[i]) * \
                       (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
        self.__rating += math.pow(self.__rd, 2) * tempSum
        
        
    def _newVol(self, rating_list, RD_list, outcome_list, v):

        i = 0
        delta = self._delta(rating_list, RD_list, outcome_list, v)
        a = math.log(math.pow(self.vol, 2))
        tau = self._tau
        x0 = a
        x1 = 0
        
        while x0 != x1:
            # New iteration, so x(i) becomes x(i-1)
            x0 = x1
            d = math.pow(self.__rating, 2) + v + math.exp(x0)
            h1 = -(x0 - a) / math.pow(tau, 2) - 0.5 * math.exp(x0) \
            / d + 0.5 * math.exp(x0) * math.pow(delta / d, 2)
            h2 = -1 / math.pow(tau, 2) - 0.5 * math.exp(x0) * \
            (math.pow(self.__rating, 2) + v) \
            / math.pow(d, 2) + 0.5 * math.pow(delta, 2) * math.exp(x0) \
            * (math.pow(self.__rating, 2) + v - math.exp(x0)) / math.pow(d, 3)
            x1 = x0 - (h1 / h2)

        return math.exp(x1 / 2)
        
    def _delta(self, rating_list, RD_list, outcome_list, v):

        tempSum = 0
        for i in range(len(rating_list)):
            tempSum += self._g(RD_list[i]) * (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
        return v * tempSum
        
    def _v(self, rating_list, RD_list):

        tempSum = 0
        for i in range(len(rating_list)):
            tempE = self._E(rating_list[i], RD_list[i])
            tempSum += math.pow(self._g(RD_list[i]), 2) * tempE * (1 - tempE)
        return 1 / tempSum
        
    def _E(self, p2rating, p2RD):

        return 1 / (1 + math.exp(-1 * self._g(p2RD) * \
                                 (self.__rating - p2rating)))
        
    def _g(self, RD):

        return 1 / math.sqrt(1 + 3 * math.pow(RD, 2) / math.pow(math.pi, 2))
        
    def did_not_compete(self):

        self._preRatingRD()