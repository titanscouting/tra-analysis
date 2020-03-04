# Titan Robotics Team 2022: Superscript Script
# Written by Arthur Lu & Jacob Levine
# Notes:
# setup:

__version__ = "0.0.1.003"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
    0.0.1.003:
        - working
    0.0.1.002:
        - started implement of metrics
    0.0.1.001:
        - cleaned up imports
    0.0.1.000:
        - tested working, can push to database
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
import data as d
import time

def testing():

    competition, config = load_config("config.csv")

    apikey = an.load_csv("keys.txt")[0][0]
    tbakey = an.load_csv("keys.txt")[1][0]

    metricsloop(tbakey, apikey, "2020mokc", 1583084980)

def main():
    while(True):
        current_time = time.time()
        print("time: " + str(current_time))

        print("loading config")
        competition, config = load_config("config.csv")
        print("config loaded")

        print("loading database keys")
        apikey = an.load_csv("keys.txt")[0][0]
        tbakey = an.load_csv("keys.txt")[1][0]
        print("loaded keys")

        print("loading data")
        data = d.get_data_formatted(apikey, competition)
        print("loaded data")

        print("running tests")
        results = simpleloop(data, config)
        print("finished tests")

        print("running metrics")
        metrics = metricsloop(apikey, competition, current_time)
        print("finished metrics")
        
        print("pushing to database")
        push_to_database(apikey, competition, results, None)
        print("pushed to database")

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

def push_to_database(apikey, competition, results, metrics):

    for team in results:

        d.push_team_tests_data(apikey, competition, team, results[team])

def metricsloop(tbakey, apikey, competition, timestamp): # listener based metrics update

    matches = d.pull_new_tba_matches(tbakey, competition, timestamp)

    red = load_metrics(apikey, competition, matches, "red")
    blu = load_metrics(apikey, competition, matches, "blue")

    elo_red_total = 0
    elo_blu_total = 0

    gl2_red_total = 0
    gl2_blu_total + 0

    for team in red:

    return

def load_metrics(apikey, competition, matches, group_name):

    for match in matches:

        for team in match[group_name]:

            group = {}

            db_data = d.get_team_metrics_data(apikey, competition, team)

            if d.get_team_metrics_data(apikey, competition, team) == None:

                elo = {"score": 1500}
                gl2 = {"score": 1500, "rd": 250, "vol": 0.06}
                ts = {"mu": 25, "sigma": 25/3}

                d.push_team_metrics_data(apikey, competition, team, {"elo":elo, "gliko2":gl2,"trueskill":ts})

                group[team] = {"elo": elo, "gl2": gl2, "ts": ts}

            else:

                metrics = db_data["metrics"]
                elo = metrics["elo"]
                gl2 = metrics["gliko2"]
                ts = metrics["trueskill"]

                group[team] = {"elo": elo, "gl2": gl2, "ts": ts}

    return group

testing()

"""
Metrics Defaults:

elo starting score = 1500
elo N = 400
elo K = 24

gl2 starting score = 1500
gl2 starting rd = 350
gl2 starting vol = 0.06
"""