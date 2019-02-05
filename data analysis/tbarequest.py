#Titan Robotics Team 2022: TBA Requests Module
#Written by Arthur Lu & Jacob Levine
#Notes:
#   this should be imported as a python module using 'import tbarequest'
#   this should be included in the local directory or environment variable
#   this module has not been optimized for multhreaded computing
#Number of easter eggs: none yet
#setup:

__version__ = "1.0.0.001"

#changelog should be viewed using print(tbarequest.__changelog__)
__changelog__ = """changelog:
1.0.1.000:
    - fixed a simple error
1.0.0.xxx:
    -added common requests and JSON processing"""
__author__ = (
    "Arthur Lu <arthurlu@ttic.edu>, "
    "Jacob Levine <jlevine@ttic.edu>,"
    )
__all__ = [
    'process_json_ret',
    'req_all_events',
    'req_event_matches',
    'req_event_insights',
    'req_event_elim_alli'
    'req_team_events',
    'req_team_matches'
    ]
#imports
import requests

#as this code is public, i'm not putting 2022's API key in here. just add it as a var in your script and go
#requests a list of events that a team went to
def req_team_events(team,year,apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/team/frc'+str(team)+'/events/'+str(year),headers=headers)
    return r

#gets every match that a team played in
def req_team_matches(team,year,apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/team/frc'+str(team)+'/matches/'+str(year), headers=headers)
    return r

#gets all events in a certain year
def req_all_events(year, apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/events/'+str(year), headers=headers)
    return r

#gets all matches for an event
def req_event_matches(event_key,apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/event/'+str(event_key)+'/matches', headers=headers)
    return r

#gets elimination alliances from a event
def req_event_elim_alli(event_key, apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/event/'+str(event_key)+'/alliances', headers=headers)
    return r

#gets TBA's insights from an event
def req_event_insights(event_key, apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/event/'+str(event_key)+'/insights', headers=headers)
    return r

#processes the json return. right now, it's slow and not great. will throw an exception if it doesn't get a good status code
def process_json_ret(req):
    if req.status_code == 200:
        keys=[]
        for i in req.json():
            for j in i.keys():
                read=False
                for k in keys:
                    if k==j:
                        read=True
                        break
                if not read:
                    keys.append(j)
        out=[]
        out.append(keys)
        for i in req.json():
            buf=[]
            for j in keys:
                try:
                    buf.append(i[j])
                except:
                    buf.append("")
            out.append(buf)
        return out
    else:
        raise ValueError('Status code is: '+req.status_code+', not 200')
