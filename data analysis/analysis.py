#Titan Robotics Team 2022: Data Analysis Module
#Written by Arthur Lu & Jacob Levine
#Notes:
#   this should be imported as a python module using 'import analysis'
#   this should be included in the local directory or environment variable
#   this module has not been optimized for multhreaded computing
#Number of easter eggs: 2

#setup:

__version__ = "1.0.6.001"

__changelog__ = """changelog:
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
    - added 1d, column, row basic stats""" #changelog should be viewed using print(analysis.__changelog__)

__author__ = (
    "Arthur Lu <arthurlu@ttic.edu>, "
    "Jacob Levine <jlevine@ttic.edu>,"
    )

__all__ = [
    '_init_device',
    'c_entities',
    'nc_entities',
    'obstacles',
    'objectives',
    'load_csv',
    'basic_stats',
    'z_score',
    'stdev_z_split',
    'histo_analysis', #histo_analysis_old is intentionally left out as it has been depreciated since v 1.0.1.005
    'poly_regression',
    'log_regression',
    'exp_regression',
    'r_squared',
    'rms',
    'calc_overfit',
    'strip_data',
    'optimize_regression',
    'basic_analysis',
    #all statistics functions left out due to integration in other functions
    ]

#now back to your regularly scheduled programming:

#imports (now in alphabetical order! v 1.0.3.006):

from bisect import bisect_left, bisect_right
import collections
import csv
from decimal import Decimal
import functools
from fractions import Fraction
from itertools import groupby
import math
import matplotlib
import numbers
import numpy as np
import random
import scipy
from sklearn import *
#import statistics <-- statistics.py functions have been integrated into analysis.py as of v 1.0.3.002
import time
import torch

class error(ValueError):

    pass

def _init_device (setting, arg): #initiates computation device for ANNs
    if setting == "cuda":
        temp = setting + ":" + str(arg)
        try:
            the_device_woman = torch.device(temp if torch.cuda.is_available() else "cpu")
            return the_device_woman #name that reference
        except:
            raise error("could not assign cuda or cpu")
    elif setting == "cpu":
        try:
            the_device_woman = torch.device("cpu")
            return the_device_woman #name that reference
        except:
            raise error("could not assign cpu")
    else:
        raise error("specified device does not exist")

class c_entities:

    c_names = []
    c_ids = []
    c_pos = []
    c_properties = []
    c_logic = []

    def debug(self):
        print("c_entities has attributes names, ids, positions, properties, and logic. __init__ takes self, 1d array of names, 1d array of ids, 2d array of positions, nd array of properties, and nd array of logic")
        return[self.c_names, self.c_ids, self.c_pos, self.c_properties, self.c_logic]
    
    def __init__(self, names, ids, pos, properties, logic):
        self.c_names = names
        self.c_ids = ids
        self.c_pos = pos
        self.c_properties = properties
        self.c_logic = logic
        return None
        

    def append(self, n_name, n_id, n_pos, n_property, n_logic):
        self.c_names.append(n_name)
        self.c_ids.append(n_id)
        self.c_pos.append(n_pos)
        self.c_properties.append(n_property)
        self.c_logic.append(n_logic)
        return None
    
    def edit(self, search, n_name, n_id, n_pos, n_property, n_logic):
        position = 0
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i
        if n_name != "null":
            self.c_names[position] = n_name
    
        if n_id != "null":
            self.c_ids[position] = n_id
    
        if n_pos != "null":
            self.c_pos[position] = n_pos
    
        if n_property != "null":
            self.c_properties[position] = n_property
    
        if n_logic != "null":
            self.c_logic[position] = n_logic
    
        return None
    
    def search(self, search):
        position = 0
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i

        return [self.c_names[position], self.c_ids[position], self.c_pos[position], self.c_properties[position], self.c_logic[position]]    

    def regurgitate(self):

        return[self.c_names, self.c_ids, self.c_pos, self.c_properties, self.c_logic]
    
class nc_entities:

    c_names = []
    c_ids = []
    c_pos = []
    c_properties = []
    c_effects = []

    def debug(self):
        print ("nc_entities (non-controlable entities) has attributes names, ids, positions, properties, and effects. __init__ takes self, 1d array of names, 1d array of ids, 2d array of positions, 2d array of properties, and 2d array of effects.")
        return[self.c_names, self.c_ids, self.c_pos, self.c_properties, self.c_effects]

    def __init__(self, names, ids, pos, properties, effects):
        self.c_names = names
        self.c_ids = ids
        self.c_pos = pos
        self.c_properties = properties
        self.c_effects = effects
        return None

    def append(self, n_name, n_id, n_pos, n_property, n_effect):
        self.c_names.append(n_name)
        self.c_ids.append(n_id)
        self.c_pos.append(n_pos)
        self.c_properties.append(n_property)
        self.c_effects.append(n_effect)
        
        return None

    def edit(self, search, n_name, n_id, n_pos, n_property, n_effect):
        position = 0
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i
        if n_name != "null":
            self.c_names[position] = n_name

        if n_id != "null":
            self.c_ids[position] = n_id

        if n_pos != "null":
            self.c_pos[position] = n_pos

        if n_property != "null":
            self.c_properties[position] = n_property

        if n_effect != "null":
            self.c_effects[position] = n_effect

        return None

    def search(self, search):
        position = 0
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i

        return [self.c_names[position], self.c_ids[position], self.c_pos[position], self.c_properties[position], self.c_effects[position]]        

    def regurgitate(self):

        return[self.c_names, self.c_ids, self.c_pos, self.c_properties, self.c_effects]

class obstacles:

    c_names = []
    c_ids = []
    c_perim = []
    c_effects = []

    def debug(self):
        print("obstacles has atributes names, ids, positions, perimeters, and effects. __init__ takes self, 1d array of names, 1d array of ids, 2d array of position, 3d array of perimeters, 2d array of effects.")
        return [self.c_names, self.c_ids, self.c_perim, self.c_effects]

    def __init__(self, names, ids, perims, effects):
        self.c_names = names
        self.c_ids = ids
        self.c_perim = perims
        self.c_effects = effects
        return None

    def append(self, n_name, n_id, n_perim, n_effect):
        self.c_names.append(n_name)
        self.c_ids.append(n_id)
        self.c_perim.append(n_perim)
        self.c_effects.append(n_effect)
        return None

    def edit(self, search, n_name, n_id, n_perim, n_effect):
        position = 0
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i

        if n_name != "null":
            self.c_names[position] = n_name

        if n_id != "null":
            self.c_ids[position] = n_id

        if n_perim != "null":
            self.c_perim[position] = n_perim

        if n_effect != "null":
            self.c_effects[position] = n_effect
            
        return None

    def search(self, search):
        position = 0
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i

        return [self.c_names[position], self.c_ids[position], self.c_perim[position], self.c_effects[position]]

    def regurgitate(self):

        return[self.c_names, self.c_ids, self.c_perim, self.c_effects]

class objectives:
    
    c_names = []
    c_ids = []
    c_pos = []
    c_effects = []

    def debug(self):
        print("objectives has atributes names, ids, positions, and effects. __init__ takes self, 1d array of names, 1d array of ids, 2d array of position, 1d array of effects.")
        return [self.c_names, self.c_ids, self.c_pos, self.c_effects]
    
    def __init__(self, names, ids, pos, effects):
        self.c_names = names
        self.c_ids = ids
        self.c_pos = pos
        self.c_effects = effects
        return None
    
    def append(self, n_name, n_id, n_pos, n_effect):
        self.c_names.append(n_name)
        self.c_ids.append(n_id)
        self.c_pos.append(n_pos)
        self.c_effects.append(n_effect)
        return None
    
    def edit(self, search, n_name, n_id, n_pos, n_effect):
        position = 0
        print(self.c_ids)
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i

        if n_name != "null":
            self.c_names[position] = n_name

        if n_id != "null":
            self.c_ids[position] = n_id

        if n_pos != "null":
            self.c_pos[position] = n_pos

        if n_effect != "null":
            self.c_effects[position] = n_effect
            
        return None
    
    def search(self, search):
        position = 0
        for i in range(0, len(self.c_ids), 1):
            if self.c_ids[i] == search:
                position = i

        return [self.c_names[position], self.c_ids[position], self.c_pos[position], self.c_effects[position]]

    def regurgitate(self):

        return[self.c_names, self.c_ids, self.c_pos, self.c_effects]
    
def load_csv(filepath):
    with open(filepath, newline = '') as csvfile:
        file_array = list(csv.reader(csvfile))
    return file_array

def basic_stats(data, method, arg): # data=array, mode = ['1d':1d_basic_stats, 'column':c_basic_stats, 'row':r_basic_stats], arg for mode 1 or mode 2 for column or row
    
    if method == 'debug':
        out = "basic_stats requires 3 args: data, mode, arg; where data is data to be analyzed, mode is an int from 0 - 2 depending on type of analysis (by column or by row) and is only applicable to 2d arrays (for 1d arrays use mode 1), and arg is row/column number for mode 1 or mode 2; function returns: [mean, median, mode, stdev, variance]"
        return out

    if method == "1d" or method == 0:

        data_t = []

        for i in range (0, len(data) - 1, 1):

            data_t.append(float(data[i]))
    
        _mean = mean(data_t)
        _median = median(data_t)
        try:
            _mode = mode(data_t)
        except:
            _mode = None
        try:
            _stdev = stdev(data_t)
            
        except:
            
            _stdev = None
        
        try:
            _variance = variance(data_t)
        except:
            _variance = None
        
        out = [_mean, _median, _mode, _stdev, _variance]

        return out
    
    elif method == "column" or method == 1:

        c_data = []
        c_data_sorted = []
        
        for i in data:
            try:
                c_data.append(float(i[arg]))
            except:
                pass
            
        _mean = mean(c_data)
        _median = median(c_data)
        try:
            _mode = mode(c_data)
        except:
            _mode = None
        try:
            _stdev = stdev(c_data)
        except:
            _stdev = None
        try:
            _variance = variance(c_data)
        except:
            _variance = None
        
        out = [_mean, _median, _mode, _stdev, _variance]

        return out

    elif method == "row" or method == 2:

        r_data = []

        for i in range(len(data[arg])):
            r_data.append(float(data[arg][i]))
        
        _mean = mean(r_data)
        _median = median(r_data)
        try:
            _mode = mode(r_data)
        except:
            _mode = None
        try:
            _stdev = stdev(r_data)
        except:
            _stdev = None
        try:
            _variance = variance(r_data)
        except:
            _variance = None
        
        out = [_mean, _median, _mode, _stdev, _variance]

        return out
    else:
        raise error("method error")
    
def z_score(point, mean, stdev): #returns z score with inputs of point, mean and standard deviation of spread
    score = (point - mean)/stdev
    return score

def stdev_z_split(mean, stdev, delta, low_bound, high_bound): #returns n-th percentile of spread given mean, standard deviation, lower z-score, and upper z-score

    z_split = []

    i = low_bound

    while True:

        z_split.append(float((1 / (stdev * math.sqrt(2 * math.pi))) * math.e ** (-0.5 * (((i - mean) / stdev) ** 2))))

        i = i + delta

        if i > high_bound:

            break

    return z_split

def histo_analysis_old(hist_data): #note: depreciated since v 1.0.1.005

    if hist_data == 'debug':
        return['lower estimate (5%)', 'lower middle estimate (25%)', 'middle estimate (50%)', 'higher middle estimate (75%)', 'high estimate (95%)', 'standard deviation', 'note: this has been depreciated']
    
    derivative = []
    for i in range(0, len(hist_data) - 1, 1):
        derivative.append(float(hist_data[i+1]) - float(hist_data[i]))
        
    derivative_sorted = sorted(derivative, key=int)
    mean_derivative = basic_stats(derivative_sorted, "1d", 0)[0]
    
    print(mean_derivative)
    stdev_derivative = basic_stats(derivative_sorted, "1d", 0)[3]

    low_bound = mean_derivative + -1.645 * stdev_derivative
    lm_bound = mean_derivative + -0.674 * stdev_derivative
    mid_bound = mean_derivative * 0 * stdev_derivative
    hm_bound = mean_derivative + 0.674 * stdev_derivative
    high_bound = mean_derivative + 1.645 * stdev_derivative

    low_est = float(hist_data[-1:][0]) + low_bound
    lm_est = float(hist_data[-1:][0]) + lm_bound
    mid_est = float(hist_data[-1:][0]) + mid_bound
    hm_est = float(hist_data[-1:][0]) + hm_bound
    high_est = float(hist_data[-1:][0]) + high_bound

    return [low_est, lm_est, mid_est, hm_est, high_est, stdev_derivative]

def histo_analysis(hist_data, delta, low_bound, high_bound):

    if hist_data == 'debug':
        return ('returns list of predicted values based on historical data; input delta for delta step in z-score and lower and higher bounds in number for standard deviations')

    derivative = []

    for i in range(0, len(hist_data) - 1, 1):
        derivative.append(float(hist_data[i + 1]) - float(hist_data [i]))

    derivative_sorted = sorted(derivative, key=int)
    mean_derivative = basic_stats(derivative_sorted,"1d", 0)[0]
    stdev_derivative = basic_stats(derivative_sorted, "1d", 0)[3]

    predictions = []
    pred_change = 0

    i = low_bound

    while True:
        
        if i > high_bound:
            break

        try:
            pred_change = mean_derivative + i * stdev_derivative
            
        except:
            
            pred_change = mean_derivative

        predictions.append(float(hist_data[-1:][0]) + pred_change)

        i = i + delta

    return predictions

def poly_regression(x, y, power):

    if x == "null": #if x is 'null', then x will be filled with integer points between 1 and the size of y

        x = []

        for i in range(len(y)):

            print(i)

            x.append(i+1)

    reg_eq = scipy.polyfit(x, y, deg = power)

    eq_str = ""

    for i in range(0, len(reg_eq), 1):

        if i < len(reg_eq)- 1:
            eq_str = eq_str + str(reg_eq[i]) + "*(z**" + str(len(reg_eq) - i - 1) + ")+"
        else:
            eq_str = eq_str + str(reg_eq[i]) + "*(z**" + str(len(reg_eq) - i - 1) + ")"

    vals = []

    for i in range(0, len(x), 1):
        z = x[i]

        exec("vals.append(" + eq_str + ")")

    _rms = rms(vals, y)

    r2_d2 = r_squared(vals, y) 

    return [eq_str, _rms, r2_d2]

def log_regression(x, y, base):

    x_fit = []

    for i in range(len(x)):

        x_fit.append(np.log(x[i]) / np.log(base)) #change of base for logs

    reg_eq = np.polyfit(x_fit, y, 1) # y = reg_eq[0] * log(x, base) + reg_eq[1]

    eq_str = str(reg_eq[0]) + "* (np.log(z) / np.log(" + str(base) +"))+" + str(reg_eq[1])

    vals = []

    for i in range(len(x)):

        z = x[i]

        exec("vals.append(" + eq_str + ")")

    _rms = rms(vals, y)

    r2_d2 = r_squared(vals, y)

    return [eq_str, _rms, r2_d2]

def exp_regression(x, y, base):

    y_fit = []

    for i in range(len(y)):
        
        y_fit.append(np.log(y[i]) / np.log(base)) #change of base for logs

    reg_eq = np.polyfit(x, y_fit, 1, w=np.sqrt(y_fit)) # y = base ^ (reg_eq[0] * x) * base ^ (reg_eq[1])

    eq_str = "(" + str(base) + "**(" + str(reg_eq[0]) + "*z))*(" + str(base) + "**(" + str(reg_eq[1]) + "))"

    vals = []

    for i in range(len(x)):

        z = x[i]

        exec("vals.append(" + eq_str + ")")

    _rms = rms(vals, y)

    r2_d2 = r_squared(vals, y)

    return [eq_str, _rms, r2_d2]
    
def r_squared(predictions, targets): # assumes equal size inputs

    out = metrics.r2_score(targets, predictions)

    return out

def rms(predictions, targets): # assumes equal size inputs

    out = 0

    _sum = 0

    avg = 0

    for i in range(0, len(targets), 1):

        _sum = (targets[i] - predictions[i]) ** 2

    avg = _sum/len(targets)

    out = math.sqrt(avg)

    return float(out)

def calc_overfit(equation, rms_train, r2_train, x_test, y_test):

    #overfit = performance(train) - performance(test) where performance is r^2
    #overfir = error(train) - error(test) where error is rms

    vals = []

    for i in range(0, len(x_test), 1):

        z = x_test[i]

        exec("vals.append(" + equation + ")")
        
    r2_test = r_squared(vals, y_test)
    rms_test = rms(vals, y_test)

    return rms_train - rms_test, r2_train - r2_test

def strip_data(data, mode):

    if mode == "adam": #x is the row number, y are the data

        pass

    if mode == "eve": #x are the data, y is the column number

        pass

    else:

        raise error("mode error")

def optimize_regression(x, y, _range, resolution):#_range in poly regression is the range of powers tried, and in log/exp it is the inverse of the stepsize taken from -1000 to 1000
#usage not: for demonstration purpose only, performance is shit
    if type(resolution) != int:

        raise error("resolution must be int")
    x = x
    y = y

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for i in range (0, math.floor(len(x) * 0.4), 1):

        index = random.randint(0, len(x) - 1)

        x_test.append(x[index])
        y_test.append(y[index])

        x.pop(index)
        y.pop(index)

    x_train = x
    y_train = y

    #print(x_train, x_test)
    #print(y_train, y_test)
    
    eqs = []

    rmss = []

    r2s = []

    for i in range (0, _range + 1, 1):

        eqs.append(poly_regression(x_train, y_train, i)[0])
        rmss.append(poly_regression(x_train, y_train, i)[1])
        r2s.append(poly_regression(x_train, y_train, i)[2])

    for i in range (1, 100 * resolution + 1):

        try:
        
            eqs.append(exp_regression(x_train, y_train, float(i / resolution))[0])
            rmss.append(exp_regression(x_train, y_train, float(i / resolution))[1])
            r2s.append(exp_regression(x_train, y_train, float(i / resolution))[2])

        except:

            pass

    for i in range (1, 100 * resolution + 1):

        try:

            eqs.append(log_regression(x_train, y_train, float(i / resolution))[0])
            rmss.append(log_regression(x_train, y_train, float(i / resolution))[1])
            r2s.append(log_regression(x_train, y_train, float(i / resolution))[2])

        except:

            pass
    
    for i in range (0, len(eqs), 1): #marks all equations where r2 = 1 as they 95% of the time overfit the data

        if r2s[i] == 1:

            eqs[i] = ""
            rmss[i] = ""
            r2s[i] = ""

    while True: #removes all equations marked for removal

        try:
        
            eqs.remove('')
            rmss.remove('')
            r2s.remove('')

        except:

            break

    overfit = []

    for i in range (0, len(eqs), 1):

        overfit.append(calc_overfit(eqs[i], rmss[i], r2s[i], x_test, y_test))
            
    return eqs, rmss, r2s, overfit

def basic_analysis(filepath): #assumes that rows are the independent variable and columns are the dependant. also assumes that time flows from lowest column to highest column.
    
    data = load_csv(filepath)
    row = len(data)
    
    column = []
    
    for i in range(0, row, 1):
        
        column.append(len(data[i]))
        
    column_max = max(column)
    row_b_stats = []
    row_histo = []
    
    for i in range(0, row, 1):
        row_b_stats.append(basic_stats(data, "row", i))
        row_histo.append(histo_analysis(data[i], 0.67449, -0.67449, 0.67449))
    
    column_b_stats = []
    
    for i in range(0, column_max, 1):
        column_b_stats.append(basic_stats(data, "column", i))
    
    return[row_b_stats, column_b_stats, row_histo]

def benchmark(x, y):

    start_g = time.time()
    generate_data("benchmark_data.csv", x, y, -10, 10)
    end_g = time.time()

    start_a = time.time()
    basic_analysis("benchmark_data.csv")
    end_a = time.time()

    return [(end_g - start_g), (end_a - start_a)]

def generate_data(filename, x, y, low, high):

    file = open(filename, "w")

    for i in range (0, y, 1):

        temp = ""
        
        for j in range (0, x - 1, 1):

            temp = str(random.uniform(low, high)) +  ","  + temp

        temp = temp + str(random.uniform(low, high))
        file.write(temp + "\n")

def debug():

    data = load_csv('data.txt')

    print("--------------------------------")

    print(basic_stats(0, 'debug', 0))
    print(basic_stats(data, "column", 0))
    print(basic_stats(data, "row", 0))
    print(z_score(10, basic_stats(data, "column", 0)[0], basic_stats(data, "column", 0)[3]))
    print(histo_analysis(data[0], 0.01, -1, 1))
    print(stdev_z_split(3.3, 0.2, 0.1, -5, 5))

    print("--------------------------------")

    game_c_entities = c_entities(["bot", "bot", "bot"], [0, 1, 2], [[10, 10], [-10, -10], [10, -10]], ["shit", "bad", "worse"], ["triangle", "square", "circle"])
    game_c_entities.append("bot", 3, [-10, 10], "useless", "pentagram")
    game_c_entities.edit(0, "null", "null", "null", "null", "triagon")
    print(game_c_entities.search(0))
    print(game_c_entities.debug())
    print(game_c_entities.regurgitate())

    print("--------------------------------")

    game_nc_entities = nc_entities(["cube", "cube", "ball"], [0, 1, 2], [[0, 0.5], [1, 1.5], [2, 2]], ["1;1;1;10', '2;1;1;20", "r=0.5, 5"], ["1", "1", "0"])
    game_nc_entities.append("cone", 3, [1, -1], "property", "effect")
    game_nc_entities.edit(2, "sphere", 10, [5, -5], "new prop", "new effect")
    print(game_nc_entities.search(10))
    print(game_nc_entities.debug())
    print(game_nc_entities.regurgitate())

    print("--------------------------------")

    game_obstacles = obstacles(["wall", "fortress", "castle"], [0, 1, 2],[[[10, 10], [10, 9], [9, 10], [9, 9]], [[-10, 9], [-10, -9], [-9, -10]], [[5, 0], [4, -1], [-4, -1]]] , [0, 0.01, 10])
    game_obstacles.append("bastion", 3, [[50, 50], [49, 50], [50, 49], [49, 49]], 75)
    game_obstacles.edit(0, "motte and bailey", "null", [[10, 10], [9, 10], [10, 9], [9, 9]], 0.01)
    print(game_obstacles.search(0))
    print(game_obstacles.debug())
    print(game_obstacles.regurgitate())

    print("--------------------------------")

    game_objectives = objectives(["switch", "scale", "climb"], [0,1,2], [[0,0],[1,1],[2,0]], ["0,1", "1,1", "0,5"])
    game_objectives.append("auto", 3, [0, 10], "1, 10")
    game_objectives.edit(3, "null", 4, "null", "null")
    print(game_objectives.search(4))
    print(game_objectives.debug())
    print(game_objectives.regurgitate())

    print("--------------------------------")

    print(poly_regression([1, 2, 3, 4, 5], [1, 2, 4, 8, 16], 2))

    print(log_regression([1, 2, 3, 4], [2, 4, 8, 16], 2.717))

    print(exp_regression([1, 2, 3, 4], [2, 4, 8, 16], 2.717))

    x, y, z = optimize_regression([0, 1, 2, 3, 4], [1, 2, 4, 7, 19], 10, 100)

    for i in range(0, len(x), 1):

        print(str(x[i]) + " | " + str(y[i]) + " | " + str(z[i]))

#statistics def below------------------------------------------------------------------------------------------------------------------------------------------------------

class StatisticsError(ValueError):
    pass

def _sum(data, start=0):
    count = 0
    n, d = _exact_ratio(start)
    partials = {d: n}
    partials_get = partials.get
    T = _coerce(int, type(start))
    for typ, values in groupby(data, type):
        T = _coerce(T, typ)  # or raise TypeError
        for n,d in map(_exact_ratio, values):
            count += 1
            partials[d] = partials_get(d, 0) + n
    if None in partials:

        total = partials[None]
        assert not _isfinite(total)
    else:

        total = sum(Fraction(n, d) for d, n in sorted(partials.items()))
    return (T, total, count)

def _isfinite(x):
    try:
        return x.is_finite()  # Likely a Decimal.
    except AttributeError:
        return math.isfinite(x)  # Coerces to float first.

def _coerce(T, S):

    assert T is not bool, "initial type T is bool"

    if T is S:  return T

    if S is int or S is bool:  return T
    if T is int:  return S

    if issubclass(S, T):  return S
    if issubclass(T, S):  return T

    if issubclass(T, int):  return S
    if issubclass(S, int):  return T

    if issubclass(T, Fraction) and issubclass(S, float):
        return S
    if issubclass(T, float) and issubclass(S, Fraction):
        return T

    msg = "don't know how to coerce %s and %s"
    raise TypeError(msg % (T.__name__, S.__name__))

def _exact_ratio(x):

    try:

        if type(x) is float or type(x) is Decimal:
            return x.as_integer_ratio()
        try:

            return (x.numerator, x.denominator)
        except AttributeError:
            try:

                return x.as_integer_ratio()
            except AttributeError:

                pass
    except (OverflowError, ValueError):

        assert not _isfinite(x)
        return (x, None)
    msg = "can't convert type '{}' to numerator/denominator"
    raise TypeError(msg.format(type(x).__name__))

def _convert(value, T):

    if type(value) is T:

        return value
    if issubclass(T, int) and value.denominator != 1:
        T = float
    try:

        return T(value)
    except TypeError:
        if issubclass(T, Decimal):
            return T(value.numerator)/T(value.denominator)
        else:
            raise

def _counts(data):

    table = collections.Counter(iter(data)).most_common()
    if not table:
        return table

    maxfreq = table[0][1]
    for i in range(1, len(table)):
        if table[i][1] != maxfreq:
            table = table[:i]
            break
    return table


def _find_lteq(a, x):

    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def _find_rteq(a, l, x):

    i = bisect_right(a, x, lo=l)
    if i != (len(a)+1) and a[i-1] == x:
        return i-1
    raise ValueError


def _fail_neg(values, errmsg='negative value'):

    for x in values:
        if x < 0:
            raise StatisticsError(errmsg)
        yield x

def mean(data):

    if iter(data) is data:
        data = list(data)
    n = len(data)
    if n < 1:
        raise StatisticsError('mean requires at least one data point')
    T, total, count = _sum(data)
    assert count == n
    return _convert(total/n, T)

def median(data):
    
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    if n%2 == 1:
        return data[n//2]
    else:
        i = n//2
        return (data[i - 1] + data[i])/2

def mode(data):
    
    table = _counts(data)
    if len(table) == 1:
        return table[0][0]
    elif table:
        raise StatisticsError(
                'no unique mode; found %d equally common values' % len(table)
                )
    else:
        raise StatisticsError('no mode for empty data')

def _ss(data, c=None):

    if c is None:
        c = mean(data)
    T, total, count = _sum((x-c)**2 for x in data)

    U, total2, count2 = _sum((x-c) for x in data)
    assert T == U and count == count2
    total -=  total2**2/len(data)
    assert not total < 0, 'negative sum of square deviations: %f' % total
    return (T, total)

def variance(data, xbar=None):

    if iter(data) is data:
        data = list(data)
    n = len(data)
    if n < 2:
        raise StatisticsError('variance requires at least two data points')
    T, ss = _ss(data, xbar)
    return _convert(ss/(n-1), T)

def stdev(data, xbar=None):

    var = variance(data, xbar)
    try:
        return var.sqrt()
    except AttributeError:
        return math.sqrt(var)