import csv

results = []
with open("Req_BM.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row[1])

print results
