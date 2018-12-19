import datetime
import csv


def toCsv(rows, file, date=True, append=False):
    if date:
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        file = str(date) + "-" + file
    mode = "a" if append else "w"
    with open(file, mode) as file:
        writer = csv.writer(file)
        writer.writerows(rows)

file = "bla.csv"
# rows = [[1,2,3], [3,5,6], [7, 8, 9]]
rows = [[0, 0, 0]]
toCsv(rows, file, False, False)
