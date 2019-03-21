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

#print(full)

quant_keys = []

list_teams = ["2022", "16", "2451"]

out = []
#temp = []
var = {}
#measured_vars_total = []

for i in range(len(full)):
    for j in range(len(full[i])):
        for key in list(full[i][j].keys()):
            
            if "Quantitative" in key:
                
                quant_keys.append(key)
                
                if full[i][j].get(key).get('teamDBRef')[5:] in list_teams:
                    
                    var = {}
                    measured_vars = []
                    
                    for k in range(len(list(full[i][j].get(key).keys()))):

                        individual_keys = list(full[i][j].get(key).keys())
                        #print(individual_keys)
                        #measured_vars.append(individual_keys[k])
                        var[individual_keys[k]] = full[i][j].get(key).get(individual_keys[k])
                        #var.append(full[i][j].get(key).get(individual_keys[k]))

                    #measured_vars_total.append(measured_vars)

            #temp.append(var)
                  
        out.append(var)

sorted_out = []
#j_list = []

for i in out:

    j_list = []

    key_list = []

    sorted_keys = sorted(i.keys())

    for j in sorted_keys:

        key_list.append(i[j])

        j_list.append(j)

    sorted_out.append(key_list)
     
#print(quant_keys)
#print(measured_vars_total)
print(sorted_out)
print(j_list)

var_index = 0
team_index = 0

big_out = []

for i in sorted_out:

    team_index = list_teams.index(sorted_out[sorted_out.index(i)][j_list.index('teamDBRef')][5:])

    #print(team_index)

    for j in range(len(i)):

        print(j)
        print(team_index)
        print(i[j])

        print(i)

        print(big_out)

        big_out[j][team_index].append(i[j])

print(big_out)

#for i in range(len(measured_vars)):
    #for j in range(len)
                    
                    
                    #for k in range(len(list(full[i][j].get(key).keys()))):
                    #    ind_keys = list(full[i][j].get(key).keys())
                    #    print(ind_keys)
                    #    print(k)
                    #    print(ind_keys[k])
                    #    print(full[i][j].get(key).get('teamDBRef')[5:])
                    #    print(list_teams.index(full[i][j].get(key).get('teamDBRef')[5:]))
                    #    print(full[i][j].get(key))
                    #    var[list_teams.index(full[i][j].get(key).get('teamDBRef')[5:])].append( full[i][j].get(key).get(ind_keys[k]) )
                    #print(var)

#print(quant_keys)
