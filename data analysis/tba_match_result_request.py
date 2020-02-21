import requests
def req_event_matches(eventkey,apikey):
    headers={'X-TBA-Auth-Key':apikey}
    r=requests.get('https://www.thebluealliance.com/api/v3/event/'+eventkey+'/matches/simple', headers=headers)
    return r
def sort_through_request(request):
    if request.status_code == 200:
        x=[]
        for i in sorted(request.json(), key=lambda i: i['actual_time']):
            x.append([[i['alliances']['red']['team_keys'], i['alliances']['blue']['team_keys']],i['winning_alliance']])
        return x
