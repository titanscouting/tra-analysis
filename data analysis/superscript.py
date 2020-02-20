# Titan Robotics Team 2022: Superscript Script
# Written by Arthur Lu & Jacob Levine
# Notes:
# setup:

__version__ = "0.0.0.005"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
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
try:
    from analysis import trueskill as Trueskill
except:
    import trueskill as Trueskill

def main():

    pass

def simpleloop(data, tests): # expects 3D array with [Team][Variable][Match]

    return_vector = []

    for team in data:

        team_vector = []

        for variable in team:

            variable_vector = []

            for test in tests:

                if(test == "basic" or test == "basic_stats" or test == 0):

                    variable_vector.append(an.basic_stats(variable))

                if(test == "histo" or test == "histo_analysis" or test == 1):

                    variable_vector.append(an.histo_analysis(variable))

                if(test == "r.lin" or test == "regression.lin" or test == 2):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["lin"]))

                if(test == "r.log" or test == "regression.log" or test == 3):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["log"]))

                if(test == "r.exp" or test == "regression.exp" or test == 4):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["exp"]))

                if(test == "r.ply" or test == "regression.ply" or test == 5):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["ply"]))

                if(test == "r.sig" or test == "regression.sig" or test == 6):

                    variable_vector.append(an.regression("cpu", range(0, len(variable) - 1), variable, ["sig"]))

            team_vector.append(variable_vector)

        return_vector.append(team_vector)

    return return_vector

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

def metricsloop(group_data, observations, database, tests): # listener based metrics update

    pass

def metricsloop_dumb(team_lookup, data, tests): # expects array with [Match] ([Teams], [Win/Loss])

    scores = []

    elo_starting_score = 1500
    N = 1500
    K = 32

    gl2_starting_score = 1500
    gl2_starting_rd = 350
    gl2_starting_vol = 0.06

    for team in team_lookup:

        elo = elo_starting_score
        gl2 = {"score": gl2_starting_score, "rd": gl2_starting_rd, "vol": gl2_starting_vol}
        ts = Trueskill.Rating()

        scores[str(team)] = {"elo": elo, "gl2": gl2, "ts": ts}

    for match in data:

        groups = data[0]

        for group in groups:

            group_vector = []

            for team in group:

                group_vector.append(scores[team])

            group_ratings.append(group_vector)

        observations = data[1]

        new_group_ratings = []

main()