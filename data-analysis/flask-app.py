import flask
import threading 

i = 0

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def heartbeat():
    return ""

@app.route('/getprogress', methods=['GET'])
def getprogress():
	return str(i)

def main():

	global i

	while(True):

		i += 1

task = threading.Thread(name = "main", target=main)
task.start()
app.run()
