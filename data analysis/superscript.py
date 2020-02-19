# Titan Robotics Team 2022: Superscript Script
# Written by Arthur Lu & Jacob Levine
# Notes:
# setup:

__version__ = "0.0.0.002"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
    0.0.0.002:
        - added simpleloop which is untested until data is provided
    0.0.0.001:
        - created script
        - added analysis, numba, numpy imports
"""

__author__ = (
    "Arthur Lu <learthurgo@gmail.com>",
    "Jacob Levine <jlevine@imsa.edu>",
)

__all__ = [
]

# imports:

from analysis import analysis as an
from numba import jit
import numpy as np

def main():

    pass

def simpleloop(data, tests): # expects 3D array with [Team][Variable][Match]

    return_vector = []

    for team in teams:

        team_vector = []

        for variable in teams:

            variable_vector = []

            for test in tests:

                if(test == "basic" or test == "basic_stats" or test == 0):

                    variable_vector.append(an.basic_stats(variable))

                if(test == "histo" or test == "histo_analysis" or test == 1):

                    variable_vector.append(an.histo_analysis(variable))

                if(test == "sr.lin" or test == "sregression.lin" or test == 2):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["lin"]))

                if(test == "sr.log" or test == "sregression.log" or test == 3):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["log"]))

                if(test == "sr.exp" or test == "sregression.exp" or test == 4):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["exp"]))

                if(test == "sr.ply" or test == "sregression.ply" or test == 5):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["ply"]))

                if(test == "sr.sig" or test == "sregression.sig" or test == 6):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["sig"]))

def metricsloop(data):

    pass

main()