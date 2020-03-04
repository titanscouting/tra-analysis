# Titan Robotics Team 2022: Superscript Script
# Written by Arthur Lu & Jacob Levine
# Notes:
# setup:

__version__ = "0.0.0.009"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
    0.0.0.009:
        - tested working
        - prints out stats for the time being, will push to database later
    0.0.0.008:
        - added data import
        - removed tba import
        - finished main method
    0.0.0.007:
        - added load_config
        - optimized simpleloop for readibility
        - added __all__ entries
        - added simplestats engine
        - pending testing
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
    "main", 
    "load_config",
    "simpleloop",
    "simplestats",
    "metricsloop"
]

# imports:

from analysis import analysis as an
from numba import jit
import numpy as np
import pickle
import data as d
try:
    from analysis import trueskill as Trueskill
except:
    import trueskill as Trueskill

def main():
    while(True):
        competition, config = load_config("config.csv")
        apikey = an.load_csv("keys.txt")[0][0]
        data = d.get_data_formatted(apikey, competition)
        results = simpleloop(data, config)
        #print(data)
        print(results)

def load_config(file):
    config_vector = {}
    file = an.load_csv(file)
    for line in file[1:]:
        config_vector[line[0]] = line[1:]

    return (file[0][0], config_vector)

def simpleloop(data, tests): # expects 3D array with [Team][Variable][Match]
    return_vector = {}
    for team in data:
        variable_vector = {}
        for variable in data[team]:
            test_vector = {}
            variable_data = data[team][variable]
            if(variable in tests):
                for test in tests[variable]:
                    test_vector[test] = simplestats(variable_data, test)
            else:
                pass      
            variable_vector[variable] = test_vector
        return_vector[team] = variable_vector

    return return_vector

def simplestats(data, test):

    if(test == "basic_stats"):
        return an.basic_stats(data)

    if(test == "historical_analysis"):
        return an.histo_analysis(data)

    if(test == "regression_linear"):
        return an.regression('cpu', list(range(len(data))), data, ['lin'])

    if(test == "regression_logarithmic"):
        return an.regression('cpu', list(range(len(data))), data, ['log'])

    if(test == "regression_exponential"):
        return an.regression('cpu', list(range(len(data))), data, ['exp'])

    if(test == "regression_polynomial"):
        return an.regression('cpu', list(range(len(data))), data, ['ply'])

    if(test == "regression_sigmoidal"):
        return an.regression('cpu', list(range(len(data))), data, ['sig'])

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