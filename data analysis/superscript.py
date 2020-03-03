# Titan Robotics Team 2022: Superscript Script
# Written by Arthur Lu & Jacob Levine
# Notes:
# setup:

__version__ = "0.0.0.006"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
    0.0.0.006:
        - fixes
    0.0.0.005:
        - imported pickle
        - created custom database object
    0.0.0.004:
        - fixed simpleloop to actually return a vector
    0.0.0.003:
        - added metricsloop which is unfinished
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
import pickle
import tba
try:
    from analysis import trueskill as Trueskill
except:
    import trueskill as Trueskill

def main():

    pass

def simpleloop(data, tests): # expects 3D array with [Team][Variable][Match]

    return_vector = {}

    for team in data:

        variable_vector = {}

        for variable in data[team]:

            test_vector = {}
            variable_data = data[team][variable]

            for test in tests[variable]:

                test_vector[test] = simplestats(variable_data, test)

               #test_vector[test] = None
            
            variable_vector[variable] = test_vector

        return_vector[team] = variable_vector

    return return_vector

def simplestats(data, test):

    if(test == "simple_stats"):

        return an.basic_stats(data)


def metricsloop(group_data, observations, database, tests): # listener based metrics update

    pass

class database:

    data = {}

    elo_starting_score = 1500
    N = 1500
    K = 32

    gl2_starting_score = 1500
    gl2_starting_rd = 350
    gl2_starting_vol = 0.06

    def __init__(self, team_lookup):
        super().__init__()

        for team in team_lookup:

            elo = elo_starting_score
            gl2 = {"score": gl2_starting_score, "rd": gl2_starting_rd, "vol": gl2_starting_vol}
            ts = Trueskill.Rating()

            data[str(team)] = {"elo": elo, "gl2": gl2, "ts": ts}            

    def get_team(self, team):

        return data[team]

    def get_elo(self, team):

        return data[team]["elo"]

    def get_gl2(self, team):

        return data[team]["gl2"]

    def get_ts(self, team):

        return data[team]["ts"]

    def set_team(self, team, ndata):

        data[team] = ndata

    def set_elo(self, team, nelo):

        data[team]["elo"] = nelo

    def set_gl2(self, team, ngl2):

        data[team]["gl2"] = ngl2

    def set_ts(self, team, nts):

        data[team]["ts"] = nts

    def save_database(self, location):

        pickle.dump(data, open(location, "wb"))

    def load_database(self, location):

        data = pickle.load(open(location, "rb"))

main()