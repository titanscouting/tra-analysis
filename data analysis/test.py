import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import csv
import numpy as np

# Use a service account
cred = credentials.Certificate('keys/keytemp.json')
#add your own key as this is public. email me for details
firebase_admin.initialize_app(cred)

db = firestore.client()

teams=db.collection('data').document('team-2022').collection("Central 2019").get()
full=[]
tms=[]
for team in teams:
    data=[]
    tms.append(team.id)
    reports=db.collection('data').document('team-2022').collection("Central 2019").document(team.id).collection("matches").get()
    for report in reports:
        data.append(db.collection('data').document('team-2022').collection("Central 2019").document(team.id).collection("matches").document(report.id).get().to_dict())
    full.append(data)

print(full)

quant_keys = []

list_teams = ["2022", "16", "2451"]

out = []
var = []

for i in range(len(full)):
    for j in range(len(full[i])):
        for key in list(full[i][j].keys()):
            if "Quantitative" in key:
                quant_keys.append(key)
                if full[i][j].get(key).get('teamDBRef')[5:] in list_teams:
                    

print(quant_keys)
