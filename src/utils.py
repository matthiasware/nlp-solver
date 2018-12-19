import csv
import datetime


def dictToHtmlTable(dctlist, fcolor):
    header = dctlist[0].keys()


def toHtmlTable(header, items):
    row = "<tr>" + len(items[0]) * "<td>{}</td>" + "</tr>"
    html = "<table cellspacing=\"0\" cellpadding=\"10\" border=\"1|0\" '>"
    html += row.format(*header)
    for i in items:
        html += row.format(*i)
    html += "</table>"
    return html


def toHtmlColorTable(items):
    assert len(items) > 0
    color_row = "<tr>" + len(items[0]) * "<td bgcolor=\"{}\">{}</td>" + "</tr>"
    html = "<table cellspacing=\"0\" cellpadding=\"10\" border=\"1|0\" '>"
    for ts in items:
        html += color_row.format(*list(sum(ts, ())))
    html += "</table>"
    return html


def toLatexColorTable(items):
    ncols = len(items[0])
    color_row = "\\cellcolor{{{}}} {}"
    color_row += "& \\cellcolor{{{}}} {}" * (ncols - 1) + "\\\\"

    latex = "\\begin{longtable}{" + ncols * "c" + "}"
    for i, ts in enumerate(items):
        latex += color_row.format(*list(sum(ts, ())))
        latex += "\n"
        if i == 0:
            latex += "\endhead"
    latex += "\\end{longtable}"
    return latex

def toHtml(*args):
    html = "<html><head></head><body>"
    for i in args:
        html += i
    html += "</body></html>"
    return html


def toHtmlString(header, items):
    row = "<tr>" + len(header) * "<td>{}</td>" + "</tr>"
    # problem_name, num_var, status_code, iter ations, f_final
    html = "<html><head></head><body><table cellspacing=\"0\" cellpadding=\"10\" border=\"1|0\" '>"
    html += row.format(*header)
    for i in items:
        html += row.format(*i)
    html += "</table></body></html>"
    return html


def toCsvFile(lst, filename, header=None, date=False):
    if date:
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        filename = str(date) + "-" + filename
    with open(filename, "w") as file:
        if header:
            lst.insert(0, header)
        writer = csv.writer(file)
        writer.writerows(lst)


def toFile(string, filename, date=False):
    if date:
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = str(date) + "-" + filename
    with open(filename, "w") as file:
        file.write(string)


def getResultsAsDict(file):
    # header row is required
    rd = {}
    with open(file, "r") as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rd[row[0]] = {h: row[i + 1] for i, h in enumerate(header[1:])}
    return rd


def zhandler(signum, frame):
    html = toHtml(failed, success)
    print(html)
    print('Exiting program: Ctrl+Z pressed')
    sys.exit(0)