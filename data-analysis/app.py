import flask
import json
import superscript as su
import threading

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

app = flask.Flask(__name__)
app.config["DEBUG"] = True

match = False
metric = False
pit = False

match_enable = True
metric_enable = True
pit_enable = True

config = {}

@app.route('/', methods=['GET'])
def heartbeat():
    return ""

@app.route('/get-match-status', methods=['GET'])
def get_match():
	return str(match)

@app.route('/get-metric-status', methods=['GET'])
def get_metric():
	return str(metric)

@app.route('/get-pit-status', methods=['GET'])
def get_pit():
	return str(pit)

@app.route('/get-match-enable', methods=['GET'])
def get_match_enable():
	return str(match_enable)

@app.route('/get-metric-enable', methods=['GET'])
def get_metric_enable():
	return str(metric_enable)

@app.route('/get-pit-enable', methods=['GET'])
def get_pit_enable():
	return str(pit_enable)

@app.route('/set-match-enable', methods=['PUT'])
def set_match_enable():
	if str(flask.request.data.get('text', '')) == "True":
		match_enable = True
	elif str(flask.request.data.get('text', '')) == "False":
		match_enable = False
	else:
		return "False"
	return "True"

@app.route('/set-metric-enable', methods=['PUT'])
def set_metric_enable():
	if str(flask.request.data.get('text', '')) == "True":
		metric_enable = True
	elif str(flask.request.data.get('text', '')) == "False":
		metric_enable = False
	else:
		return "False"
	return "True"

@app.route('/set-pit-enable', methods=['PUT'])
def set_pit_enable():
	if str(flask.request.data.get('text', '')) == "True":
		pit_enable = True
	elif str(flask.request.data.get('text', '')) == "False":
		pit_enable = False
	else:
		return "False"
	return "True"

@app.route('/get-config', methods=['GET'])
def get_config():
	return config

@app.route('/set-config', methods=['PUT'])
def set_config():
	config = json.loads(str(flask.request.data.get('text', '')))
	su.save_config("config.json", config)
	return "True"

def main():

	global match
	global metric
	global pit

	global match_enable
	global metric_enable
	global pit_enable

	global config
	config = su.load_config("config.json")

	while(True):

		if match_enable == True and match == False:

			def target():

				apikey = config["key"]["database"]
				competition = config["competition"]
				tests = config["statistics"]["match"]

				data = su.load_match(apikey, competition)
				su.matchloop(apikey, competition, data, tests)

				match = False
				return

			match = True
			task = threading.Thread(name = "match", target=target)
			task.start()

		if metric_enable == True and metric == False:
			
			def target():

				apikey = config["key"]["database"]
				tbakey = config["key"]["tba"]
				competition = config["competition"]
				metric = config["statistics"]["metric"]

				timestamp = su.get_previous_time(apikey)

				su.metricloop(tbakey, apikey, competition, timestamp, metric)

				metric = False
				return

			match = True
			task = threading.Thread(name = "metric", target=target)
			task.start()

		if pit_enable == True and pit == False:

			def target():

				apikey = config["key"]["database"]
				competition = config["competition"]
				tests = config["statistics"]["pit"]

				data = su.load_pit(apikey, competition)
				su.pitloop(apikey, competition, data, tests)

				pit = False
				return

			pit = True
			task = threading.Thread(name = "pit", target=target)
			task.start()
			
task = threading.Thread(name = "main", target=main)
task.start()
app.run()