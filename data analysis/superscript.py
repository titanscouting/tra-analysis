# Titan Robotics Team 2022: Superscript Script
# Written by Arthur Lu & Jacob Levine
# Notes:
# setup:

__version__ = "0.0.3.000"

# changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
    0.0.3.00:
        - added analysis to pit data
    0.0.2.001:
        - minor stability patches
        - implemented db syncing for timestamps
        - fixed bugs
    0.0.2.000:
        - finalized testing and small fixes
    0.0.1.004:
        - finished metrics implement, trueskill is bugged
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
import matplotlib.pyplot as plt
import time
import warnings

def main():
    warnings.filterwarnings("ignore")
    while(True):

        current_time = time.time()
        print("time: " + str(current_time))

        print(" loading config")
        competition, config = load_config("config.csv")
        print(" config loaded")

        print(" loading database keys")
        apikey = an.load_csv("keys.txt")[0][0]
        tbakey = an.load_csv("keys.txt")[1][0]
        print(" loaded keys")

        previous_time = d.get_analysis_flags(apikey, "latest_update")

        if(previous_time == None):

            d.set_analysis_flags(apikey, "latest_update", 0)
            previous_time = 0

        else:

            previous_time = previous_time["latest_update"]

        print(" analysis backtimed to: " + str(previous_time))

        print(" loading data")
        data = d.get_match_data_formatted(apikey, competition)
        pit_data = d.pit = d.get_pit_data_formatted(apikey, competition)
        print(" loaded data")

        print(" running tests")
        results = simpleloop(data, config)
        print(" finished tests")

        print(" running metrics")
        metrics = metricsloop(tbakey, apikey, competition, previous_time)
        print(" finished metrics")

        print(" running pit analysis")
        pit = pitloop(pit_data, config)
        print(" finished pit analysis")

        d.set_analysis_flags(apikey, "latest_update", {"latest_update":current_time})
        
        print(" pushing to database")
        push_to_database(apikey, competition, results, metrics, pit)
        print(" pushed to database")

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
        return an.histo_analysis([list(range(len(data))), data])

    if(test == "regression_linear"):
        return an.regression(list(range(len(data))), data, ['lin'])

    if(test == "regression_logarithmic"):
        return an.regression(list(range(len(data))), data, ['log'])

    if(test == "regression_exponential"):
        return an.regression(list(range(len(data))), data, ['exp'])

    if(test == "regression_polynomial"):
        return an.regression(list(range(len(data))), data, ['ply'])

    if(test == "regression_sigmoidal"):
        return an.regression(list(range(len(data))), data, ['sig'])

def push_to_database(apikey, competition, results, metrics, pit):

    for team in results:

        d.push_team_tests_data(apikey, competition, team, results[team])

    for team in metrics:

        d.push_team_metrics_data(apikey, competition, team, metrics[team])

    for variable in pit:

        d.push_team_pit_data(apikey, competition, variable, pit[variable])

def metricsloop(tbakey, apikey, competition, timestamp): # listener based metrics update

    elo_N = 400
    elo_K = 24

    matches = d.pull_new_tba_matches(tbakey, competition, timestamp)

    return_vector = {}

    red = {}
    blu = {}

    for match in matches:

        red = load_metrics(apikey, competition, match, "red")
        blu = load_metrics(apikey, competition, match, "blue")

        elo_red_total = 0
        elo_blu_total = 0

        gl2_red_score_total = 0
        gl2_blu_score_total = 0

        gl2_red_rd_total = 0
        gl2_blu_rd_total = 0

        gl2_red_vol_total = 0
        gl2_blu_vol_total = 0

        for team in red:

            elo_red_total += red[team]["elo"]["score"]

            gl2_red_score_total += red[team]["gl2"]["score"]
            gl2_red_rd_total += red[team]["gl2"]["rd"]
            gl2_red_vol_total += red[team]["gl2"]["vol"]

        for team in blu:

            elo_blu_total += blu[team]["elo"]["score"]

            gl2_blu_score_total += blu[team]["gl2"]["score"]
            gl2_blu_rd_total += blu[team]["gl2"]["rd"]
            gl2_blu_vol_total += blu[team]["gl2"]["vol"]

        red_elo = {"score": elo_red_total / len(red)}
        blu_elo = {"score": elo_blu_total / len(blu)}

        red_gl2 = {"score": gl2_red_score_total / len(red), "rd": gl2_red_rd_total / len(red), "vol": gl2_red_vol_total / len(red)}
        blu_gl2 = {"score": gl2_blu_score_total / len(blu), "rd": gl2_blu_rd_total / len(blu), "vol": gl2_blu_vol_total / len(blu)}


        if(match["winner"] == "red"):

            observations = {"red": 1, "blu": 0}

        elif(match["winner"] == "blue"):

            observations = {"red": 0, "blu": 1}

        else:

            observations = {"red": 0.5, "blu": 0.5}

        red_elo_delta = an.elo(red_elo["score"], blu_elo["score"], observations["red"], elo_N, elo_K) - red_elo["score"]
        blu_elo_delta = an.elo(blu_elo["score"], red_elo["score"], observations["blu"], elo_N, elo_K) - blu_elo["score"]

        new_red_gl2_score, new_red_gl2_rd, new_red_gl2_vol = an.glicko2(red_gl2["score"], red_gl2["rd"], red_gl2["vol"], [blu_gl2["score"]], [blu_gl2["rd"]], [observations["red"], observations["blu"]])
        new_blu_gl2_score, new_blu_gl2_rd, new_blu_gl2_vol = an.glicko2(blu_gl2["score"], blu_gl2["rd"], blu_gl2["vol"], [red_gl2["score"]], [red_gl2["rd"]], [observations["blu"], observations["red"]])

        red_gl2_delta = {"score": new_red_gl2_score - red_gl2["score"], "rd": new_red_gl2_rd - red_gl2["rd"], "vol": new_red_gl2_vol - red_gl2["vol"]}
        blu_gl2_delta = {"score": new_blu_gl2_score - blu_gl2["score"], "rd": new_blu_gl2_rd - blu_gl2["rd"], "vol": new_blu_gl2_vol - blu_gl2["vol"]}

        for team in red:

            red[team]["elo"]["score"] = red[team]["elo"]["score"] + red_elo_delta

            red[team]["gl2"]["score"] = red[team]["gl2"]["score"] + red_gl2_delta["score"]
            red[team]["gl2"]["rd"] = red[team]["gl2"]["rd"] + red_gl2_delta["rd"]
            red[team]["gl2"]["vol"] = red[team]["gl2"]["vol"] + red_gl2_delta["vol"]

        for team in blu:

            blu[team]["elo"]["score"] = blu[team]["elo"]["score"] + blu_elo_delta

            blu[team]["gl2"]["score"] = blu[team]["gl2"]["score"] + blu_gl2_delta["score"]
            blu[team]["gl2"]["rd"] = blu[team]["gl2"]["rd"] + blu_gl2_delta["rd"]
            blu[team]["gl2"]["vol"] = blu[team]["gl2"]["vol"] + blu_gl2_delta["vol"]

        """ not functional for now
        red_trueskill = []
        blu_trueskill = []

        red_ts_team_lookup = []
        blu_ts_team_lookup = []

        for team in red:

            red_trueskill.append((red[team]["ts"]["mu"], red[team]["ts"]["sigma"]))
            red_ts_team_lookup.append(team)

        for team in blu:

            blu_trueskill.append((blu[team]["ts"]["mu"], blu[team]["ts"]["sigma"]))
            blu_ts_team_lookup.append(team)

        print(red_trueskill)
        print(blu_trueskill)

        results = an.trueskill([red_trueskill, blu_trueskill], [observations["red"], observations["blu"]])

        print(results)

        """

    return_vector.update(red)
    return_vector.update(blu)

    return return_vector

def load_metrics(apikey, competition, match, group_name):

    group = {}

    for team in match[group_name]:

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
            gl2 = metrics["gl2"]
            ts = metrics["trueskill"]

            group[team] = {"elo": elo, "gl2": gl2, "ts": ts}

    return group

def pitloop(pit, tests):

    return_vector = {}
    for team in pit:
        for variable in pit[team]:
            if(variable in tests):
                if(not variable in return_vector):
                    return_vector[variable] = []
                return_vector[variable].append(pit[team][variable])

    return return_vector

main()

"""
Metrics Defaults:

elo starting score = 1500
elo N = 400
elo K = 24

gl2 starting score = 1500
gl2 starting rd = 350
gl2 starting vol = 0.06
"""