import utils
from run_cutest import toCsv

f1 = "master.csv"
f2 = "2018-02-23 13:02-master-soeren-.csv"


csvdict1 = utils.getResultsAsDict(f1)
csvdict2 = utils.getResultsAsDict(f2)

for key in csvdict2.keys():
    csvdict1[key] = csvdict2[key]

header = ["name", "bounds", "n", "m",
          "success", "status", "iter", "f", "tsec", "msg"]
rows = []
for k, v in csvdict1.items():
    row = [k]
    for h in header[1:]:
        row.append(v[h])
    rows.append(row)

rows = sorted(rows, key=lambda r: r[0])
rows.insert(0, header)
toCsv(rows, "test.csv", date=False)
