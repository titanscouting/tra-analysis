import os
import json
import ordereddict
import collections
import unicodecsv

content = open("data/realtimeDatabaseExport2018.json").read()

dict_content = json.loads(content)
list_of_new_data = []

for datak, datav in dict_content.iteritems():
    for teamk, teamv in datav["teams"].iteritems():
        for matchk, matchv in teamv.iteritems():
            for detailk, detailv in matchv.iteritems():
                new_data = collections.OrderedDict(detailv)
                new_data["uuid"] = detailk
                new_data["match"] = matchk
                new_data["team"] = teamk
                
                list_of_new_data.append(new_data)

allkey = reduce(lambda x, y: x.union(y.keys()), list_of_new_data, set())
output_file = open('realtimeDatabaseExport2018.csv', 'wb')
dict_writer = unicodecsv.DictWriter(csvfile=output_file, fieldnames=allkey)
dict_writer.writerow(dict((fn,fn) for fn in dict_writer.fieldnames))
dict_writer.writerows(list_of_new_data)
output_file.close()
