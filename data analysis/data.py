import requests
import pymongo
import pandas as pd
def req_event_matches(eventkey,apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/event/'+eventkey+'/matches/simple', headers=headers)
    return r
def get_match_data(request):
    if request.status_code == 200:
        x=[]
        for i in sorted(request.json(), key=lambda i: i['actual_time']):
            x.append([[i['alliances']['red']['team_keys'], i['alliances']['blue']['team_keys']],i['winning_alliance']])
        return x

def get_team_match_data(apikey, competition, team_num):
    client = pymongo.MongoClient(apikey)
    db = client.data_scouting
    mdata = db.matchdata
    out = {}
    for i in mdata.find({"competition" : competition, "team_scouted": team_num}):
        out[i['match']] = i['data']
    return pd.DataFrame(out)

def get_team_pit_data(apikey, competition, team_num):
    client = pymongo.MongoClient(apikey)
    db = client.data_scouting
    mdata = db.pitdata
    out = {}
    for i in mdata.find({"competition" : competition, "team_scouted": team_num}):
        out[i['match']] = i['data']
    return pd.DataFrame(out)

def unkeyify_2l(layered_dict):
    out = {}
    for i in layered_dict.keys():
        add = []
        sortkey = []
        for j in layered_dict[i].keys():
            add.append([j,layered_dict[i][j]])
        add.sort(key = lambda x: x[0])
        out[i] = list(map(lambda x: x[1], add))
    return out

def get_data_formatted(apikey, competition):
    client = pymongo.MongoClient(apikey)
    db = client.data_scouting
    mdata = db.teamlist
    x=mdata.find_one({"competition":competition})
    out = {}
    for i in x:
        try:
            out[int(i)] = unkeyify_2l(get_team_match_data(apikey, competition, int(i)).transpose().to_dict())
        except:
            pass
    return out

def push_team_data(apikey, competition, team_num, data):
    client = pymongo.MongoClient(apikey)
    db = client.data_processing
    mdata = db.team_tests
    mdata.replace_one({"competition" : competition, "team": team_num}, {"_id": competition+str(team_num)+"am", "competition" : competition, "team" : team_num, "data" : data}, True)