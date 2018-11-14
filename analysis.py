#Titan Robotics Team 2022: Data Analysis Module
#Written by Arthur Lu & Jacob Levine
#Notes:
#   this should be imported as a python module using 'import analysis'
#   this should be included in the local directory or environment variable
#   this module has not been optimized for multhreaded computing
#Number of easter eggs: 2

#setup:

__version__ = "1.0.3.001"

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
    'histo_analysis', #histo_analysis_old is intentionally left out as it has been depreciated
    'poly_regression',
    'r_squared',
    'rms',
    'basic_analysis',
    ]

#now back to your regularly scheduled programming:

import statistics
import math
import csv
import functools
import numpy as np
import time
import torch
import scipy
import matplotlib
from sklearn import *

def _init_device (setting, arg): #initiates computation device for ANNs
    if setting == "cuda":
        temp = setting + ":" + arg
        the_device_woman = torch.device(temp if torch.cuda.is_available() else "cpu")
        return the_device_woman #name that reference
    elif setting == "cpu":
        the_device_woman = torch.device("cpu")
        return the_device_woman #name that reference
    else:
        return "error:specified device does not exist"

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

def basic_stats(data, mode, arg): # data=array, mode = ['1d':1d_basic_stats, 'column':c_basic_stats, 'row':r_basic_stats], arg for mode 1 or mode 2 for column or row

    if mode == 'debug':
        out = "basic_stats requires 3 args: data, mode, arg; where data is data to be analyzed, mode is an int from 0 - 2 depending on type of analysis (by column or by row) and is only applicable to 2d arrays (for 1d arrays use mode 1), and arg is row/column number for mode 1 or mode 2; function returns: [mean, median, mode, stdev, variance]"
        return out

    if mode == "1d" or mode == 0:

        data_t = []

        for i in range (0, len(data) - 1, 1):

            data_t.append(float(data[i]))
    
        mean = statistics.mean(data_t)
        median = statistics.median(data_t)
        try:
            mode = statistics.mode(data_t)
        except:
            mode = None
        try:
            stdev = statistics.stdev(data)
            
        except:
            
            stdev = None
        
        try:
            variance = statistics.variance(data_t)
        except:
            variance = None
        
        out = [mean, median, mode, stdev, variance]

        return out
    
    elif mode == "column" or mode == 1:

        c_data = []
        c_data_sorted = []
        
        for i in data:
            try:
                c_data.append(float(i[arg]))
            except:
                pass
            
        mean = statistics.mean(c_data)
        median = statistics.median(c_data)
        try:
            mode = statistics.mode(c_data)
        except:
            mode = None
        try:
            stdev = statistics.stdev(c_data)
        except:
            stdev = None
        try:
            variance = statistics.variance(c_data)
        except:
            variance = None
        
        out = [mean, median, mode, stdev, variance]

        return out

    elif mode == "row" or mode == 2:

        r_data = []

        for i in range(len(data[arg])):
            r_data.append(float(data[arg][i]))
        
        mean = statistics.mean(r_data)
        median = statistics.median(r_data)
        try:
            mode = statistics.mode(r_data)
        except:
            mode = None
        try:
            stdev = statistics.stdev(r_data)
        except:
            stdev = None
        try:
            variance = statistics.variance(r_data)
        except:
            variance = None
        
        out = [mean, median, mode, stdev, variance]

        return out
    else:
        return ["mode_error", "mode_error"]
    
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

def histo_analysis_old(hist_data): #note: depreciated

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

    if x == "null":

        x = []

        for i in range(len(y)):

            x.append(i)

    reg_eq = scipy.polyfit(x, y, deg = power)

    print(reg_eq)

    eq_str = ""

    for i in range(0, len(reg_eq), 1):

        if i < len(reg_eq)- 1:
            eq_str = eq_str + str(reg_eq[i]) + "*(z**" + str(len(reg_eq) - i - 1) + ")+"
        else:
            eq_str = eq_str + str(reg_eq[i]) + "*(z**" + str(len(reg_eq) - i - 1) + ")"

    vals = []

    for i in range(0, len(x), 1):
        print(x[i])
        z = x[i]

        exec("vals.append(" + eq_str + ")")

    print(vals)

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
