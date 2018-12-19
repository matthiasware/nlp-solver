import utils
# import os
import pycutest
import traceback
# csv must contain headers of this form
# name,success,bounds,m,n,iter,f,msg


class RTCreator:

    def build_rows(self, files, problems=None, problem_header=None,
                   result_header=None, out="html"):
        self.out = out
        self.default_colors = ["#E3F2FD", "#BBDEFB"]
        self.header_color = "#e6e6e6"
        self.colors = {
            "best": "#00e600",
            "ok": "#80ff80",
            "err": "#ff8080",
            "poor": "#ffb84d"
        }
        if self.out == "latex":
            self.default_colors = ["default1", "default2"]
            self.header_color = "header"
            self.colors = {
                "best": "best",
                "ok": "ok",
                "err": "err",
                "poor": "poor"
            }
        self.err_eps = 1e-4
        self.eps_iter = 1
        self.eps_tsec = 1
        self.floatstr = "{0:.6E}"
        self.files = files
        self.csv_dicts = [(name, utils.getResultsAsDict(file))
                          for name, file in files]
        self.solver = [i[0] for i in self.csv_dicts]
        if problems is None:
            keys = [i for d in self.csv_dicts for i in d[1].keys()]
            problems = sorted(list(set(keys)))
        self.problems = problems
        if problem_header is None:
            problem_header = ["name", "bounds", "n", "m"]
        self.problem_header = problem_header
        if result_header is None:
            result_header = ["success",
                             "f",
                             "iter",
                             "tsec",
                             "msg"
                             ]
        self.result_header = result_header
        self.rfuncmap = {"success": self.calc_success,
                         "f": self.calc_f,
                         "iter": self.calc_iter,
                         "tsec": self.calc_tsec,
                         "msg": self.calc_msg}
        rows = []
        header = self.build_header()
        if self.out == "latex":
            rows = [header[:]]
        for i, problem in enumerate(self.problems):
            if i % 10 == 0 and self.out == "html":
                rows.append(header)
            rows.append(self.build_row(problem, i))
        if self.out == "html":
            html_table = utils.toHtmlColorTable(rows)
            result_string = utils.toHtml(html_table)
            utils.toFile(result_string, "table.html")
        if self.out == "latex":
            latex_table = utils.toLatexColorTable(rows)
            utils.toFile(latex_table, "table.tex")
            result_string = latex_table
        return result_string

    def build_header(self):
        row = []
        for h in self.problem_header:
            row.append((self.header_color, h))
        for h in self.result_header:
            for s in self.solver:
                row.append((self.header_color, h + "." + s))
            if self.out == "html":
                row.append((self.header_color, "."))
        return row

    def build_row(self, problem, i):
        color = self.default_colors[i % 2]
        row = []
        problem_properties = self.getProblem(problem)
        # create problem properties
        for h in self.problem_header:
            row.append((color, problem_properties[h]))
        for h in self.result_header:
            row += self.rfuncmap[h](problem, color)
        return row

    def getProblem(self, problem):
        for solver, dct in self.csv_dicts:
            if problem in dct:
                d = dct[problem]
                d["name"] = problem
                return d
        else:
            raise ValueError

    def calc_success(self, problem, color):
        row = []
        for solver, dct in self.csv_dicts:
            if problem in dct:
                row.append((color, dct[problem]["success"]))
            else:
                row.append((color, None))
        if self.out == "html":
            row.append((self.header_color, ""))
        return row

    def calc_msg(self, problem, color):
        row = []
        for solver, dct in self.csv_dicts:
            if problem in dct:
                row.append((color, dct[problem]["msg"]))
            else:
                row.append((color, None))
        if self.out == "html":
            row.append((self.header_color, ""))
        return row

    def calc_f(self, problem, color):
        fs = []
        for solver, dct in self.csv_dicts:
            if problem in dct:
                f = dct[problem]["f"]
                try:
                    f = float(f)
                    # f = self.floatstr.format(f)
                except ValueError:
                    print(solver, ": ", problem, ": ", f)
                    tb = traceback.format_exc()
                    print(tb)
                    f = None
            else:
                f = None
            fs.append(f)
        row = [(self.colors["err"], None)] * len(fs)
        indices = [i for i, f in enumerate(fs) if f is not None]
        if indices:
            minimum = min(fs[i] for i in indices)
            rel_errs = [(fs[i] - minimum) / (abs(minimum) + 1)
                        for i in indices]
            for rel_err, i in zip(rel_errs, indices):
                if rel_err == 0:
                    row[i] = (self.colors["best"], self.floatstr.format(fs[i]))
                elif rel_err < self.err_eps:
                    row[i] = (self.colors["ok"], self.floatstr.format(fs[i]))
                else:
                    row[i] = (self.colors["poor"], self.floatstr.format(fs[i]))
        if self.out == "html":
            row.append((self.header_color, ""))
        return row

    def calc_tsec(self, problem, color):
        secs = []
        for solver, dct in self.csv_dicts:
            if problem in dct:
                tsec = dct[problem]["tsec"]
                try:
                    tsec = float(tsec)
                except ValueError:
                    print("tsec", solver, ": ", problem, ": ", tsec)
                    tb = traceback.format_exc()
                    print(tb)
                    tsec = None
            else:
                tsec = None
            secs.append(tsec)
        row = [(self.colors["err"], None)] * len(secs)
        indices = [i for i, tsec in enumerate(secs) if tsec is not None]
        if indices:
            minimum = min(secs[i] for i in indices)
            rel_errs = [(secs[i] - minimum) / (abs(minimum) + 1)
                        for i in indices]
            for rel_err, i in zip(rel_errs, indices):
                if rel_err == 0:
                    row[i] = (self.colors["best"],
                              self.floatstr.format(secs[i]))
                elif rel_err < self.eps_tsec:
                    row[i] = (self.colors["ok"], self.floatstr.format(secs[i]))
                else:
                    row[i] = (self.colors["poor"],
                              self.floatstr.format(secs[i]))
        if self.out == "html":
            row.append((self.header_color, ""))
        return row

    def calc_iter(self, problem, color):
        its = []
        for solver, dct in self.csv_dicts:
            if problem in dct:
                it = dct[problem]["iter"]
                try:
                    it = int(it)
                except ValueError:
                    print(solver, ": ", problem, ": ", )
                    it = None
            else:
                it = None
            its.append(it)
        row = [(self.colors["err"], None)] * len(its)
        indices = [i for i, it in enumerate(its) if it is not None]
        if indices:
            minimum = min(its[i] for i in indices)
            rel_errs = [(its[i] - minimum) / (abs(minimum) + 1)
                        for i in indices]
            for rel_err, i in zip(rel_errs, indices):
                if rel_err == 0:
                    row[i] = (self.colors["best"], its[i])
                elif rel_err < self.eps_iter:
                    row[i] = (self.colors["ok"], its[i])
                else:
                    row[i] = (self.colors["poor"], its[i])
        if self.out == "html":
            row.append((self.header_color, ""))
        return row


files = [("m", "master_tex.csv"),
         ("i", "ipopt_tex.csv"),
         # ("geno", "geno.csv"),
         # ("scipy", "scipy.bfgs.csv")
         ]

problem_header = ["name",
                  "bounds",
                  "n",
                  # "m"
                  ]

result_header = ["success",
                 "f",
                 "iter",
                 # "tsec",
                 "msg"
                 ]


def table_unsolved_ipopt_master():
    names = ['EIGENBLS', 'EIGENCLS', 'FLETCBV3', 'FLETCHBV', 'GENHUMPS',
             'NONCVXU2', 'NONMSQRT', 'PALMER5A', 'PALMER7A', 'PALMER7E',
             'POWELLBC', 'RAYBENDL', 'RAYBENDS']
    pyc = pycutest.Pycutest()
    problems = pyc.getProblems(names, aslist=True)
    problems = [p.problem_name for p in problems]
    rtc = RTCreator()
    rows = rtc.build_rows(files=files,
                          problem_header=problem_header,
                          result_header=result_header,
                          problems=problems)
    utils.toFile(rows, "result_unsolved.html", date=False)


def table_solved_ipopt_master():
    names = ['3PK', 'AIRCRFTB', 'AKIVA', 'ALLINIT', 'ALLINITU', 'ARGLINA',
             'ARGLINB', 'ARGLINC', 'ARGTRIGLS', 'ARWHEAD', 'BA-L1LS', 'BA-L1SPLS',
             'BARD', 'BDQRTIC', 'BEALE', 'BENNETT5LS', 'BIGGS3', 'BIGGS5', 'BIGGS6',
             'BOX2', 'BOX3', 'BOXBODLS', 'BQP1VAR', 'BQPGABIM', 'BQPGASIM', 'BRKMCC',
             'BROWNAL', 'BROWNBS', 'BROWNDEN', 'BROWNDENE', 'BROYDN3DLS', 'BROYDN7D',
             'BROYDNBDLS', 'BRYBND', 'CAMEL6', 'CHEBYQAD', 'CHNROSNB', 'CHNRSNBM',
             'CLIFF', 'CRAGGLVY', 'CUBE', 'DANWOODLS', 'DECONVU', 'DENSCHNA',
             'DENSCHNB', 'DENSCHNC', 'DENSCHND', 'DENSCHNE', 'DENSCHNF', 'DIXMAANA',
             'DIXMAANB', 'DIXMAANC', 'DIXMAAND', 'DIXMAANE', 'DIXMAANF', 'DIXMAANG',
             'DIXMAANH', 'DJTL', 'DQDRTIC', 'ECKERLE4LS', 'EDENSCH', 'EG1', 'EG2',
             'ENGVAL1', 'ENGVAL2', 'ENSOLS', 'ERRINROS', 'ERRINRSM', 'EXPFIT', 'EXPLIN',
             'EXPLIN2', 'EXTROSNB', 'FBRAIN2LS', 'FBRAINLS', 'FLETCBV2', 'FLETCHCR',
             'FREUROTH', 'GAUSS1LS', 'GAUSS2LS', 'GAUSS3LS', 'GAUSSIAN', 'GBRAINLS',
             'GENROSE', 'GENROSEB', 'GROWTHLS', 'GULF', 'HADAMALS', 'HAIRY', 'HARKERP2',
             'HART6', 'HATFLDA', 'HATFLDB', 'HATFLDC', 'HATFLDD', 'HATFLDE', 'HATFLDFL',
             'HEART6LS', 'HEART8LS', 'HELIX', 'HIELOW', 'HILBERTA', 'HILBERTB', 'HIMMELBB',
             'HIMMELBG', 'HIMMELBH', 'HIMMELP1', 'HOLMES', 'HS1', 'HS110', 'HS2', 'HS25',
             'HS3', 'HS38', 'HS3MOD', 'HS4', 'HS45', 'HS5', 'HUMPS', 'INTEQNELS', 'JENSMP',
             'KOEBHELB', 'KOWOSB', 'LANCZOS1LS', 'LANCZOS2LS', 'LANCZOS3LS', 'LIARWHD', 'LINVERSE',
             'LOGHAIRY', 'LOGROS', 'LSC1LS', 'LUKSAN11LS', 'LUKSAN12LS', 'LUKSAN13LS', 'LUKSAN14LS',
             'LUKSAN15LS', 'LUKSAN16LS', 'LUKSAN17LS', 'LUKSAN21LS', 'LUKSAN22LS', 'MANCINO',
             'MARATOSB', 'MAXLIKA', 'MCCORMCK', 'MDHOLE', 'MEXHAT', 'MGH17LS', 'MINSURF',
             'MISRA1ALS', 'MISRA1BLS', 'MISRA1CLS', 'MISRA1DLS', 'MOREBV', 'MSQRTALS',
             'MSQRTBLS', 'NCB20B', 'NELSONLS', 'NONSCOMP', 'OSBORNEA', 'OSBORNEB', 'OSLBQP',
             'PALMER1', 'PALMER1A', 'PALMER1B', 'PALMER1C', 'PALMER1D', 'PALMER1E', 'PALMER2',
             'PALMER2A', 'PALMER2B', 'PALMER2C', 'PALMER2E', 'PALMER3', 'PALMER3A', 'PALMER3B',
             'PALMER3C', 'PALMER4', 'PALMER4A', 'PALMER4B', 'PALMER4C', 'PALMER4E', 'PALMER5C',
             'PALMER5D', 'PALMER6A', 'PALMER8A', 'PALMER8E', 'PARKCH', 'PENALTY2', 'PENTDI',
             'PFIT1LS', 'PFIT2LS', 'PFIT3LS', 'PFIT4LS', 'POWELLBSLS', 'POWELLSG', 'PROBPENL',
             'PSPDOC', 'QUDLIN', 'RAT43LS', 'ROSENBR', 'S308', 'S368', 'SANTALS', 'SBRYBND',
             'SCHMVETT', 'SENSORS', 'SIM2BQP', 'SIMBQP', 'SINEALI', 'SINEVAL', 'SISSER', 'SNAIL',
             'SPECAN', 'SROSENBR', 'SSBRYBND', 'STRATEC', 'TESTQUAD', 'THURBERLS', 'TOINTGOR',
             'TOINTGSS', 'TOINTPSP', 'TOINTQOR', 'TQUARTIC', 'TRIDIA', 'VAREIGVL', 'VESUVIOLS',
             'VESUVIOULS', 'VIBRBEAM', 'WATSON', 'WEEDS', 'WOODS', 'YATP1LS', 'YATP2LS',
             'YFIT', 'YFITU', 'ZANGWIL2']

    pyc = pycutest.Pycutest()
    problems = pyc.getProblems(names, aslist=True)
    problems = [p.problem_name for p in problems]
    rtc = RTCreator()
    rows = rtc.build_rows(files=files,
                          problem_header=problem_header,
                          result_header=result_header,
                          problems=problems)
    utils.toFile(rows, "result_solved.html", date=False)


def table_solved_ipopt():
    names = ['BDEXP', 'BIGGSB1', 'BLEACHNG', 'BQPGAUSS', 'CHAINWOO',
             'CHENHARK', 'CHWIRUT1LS', 'CHWIRUT2LS', 'DIXMAANI', 'DIXMAANJ',
             'DIXMAANK', 'DIXMAANL', 'DIXMAANM', 'DIXMAANN', 'DIXMAANO',
             'DIXMAANP', 'DQRTIC', 'DRCAV1LQ', 'DRCAV2LQ', 'DRCAV3LQ',
             'EXPQUAD', 'FLETBV3M', 'HAHN1LS', 'HIMMELBF', 'HYDC20LS',
             'INDEF', 'JIMACK', 'KIRBY2LS', 'LSC2LS', 'MEYER3',
             'MGH09LS', 'MGH10LS', 'NONCVXUN', 'NONDIA', 'NONDQUAR',
             'PALMER3E', 'PALMER5B', 'PALMER6C', 'PALMER6E', 'PALMER7C',
             'PALMER8C', 'PENALTY1', 'QR3DLS', 'QRTQUAD', 'QUARTC',
             'RAT42LS', 'ROSZMAN1LS', 'SCOSINE', 'SINQUAD', 'SPARSINE',
             'SPMSRTLS', 'SSCOSINE', 'VARDIM', 'VESUVIALS', 'WALL10']
    pyc = pycutest.Pycutest()
    problems = pyc.getProblems(names, aslist=True)
    problems = [p.problem_name for p in problems]
    rtc = RTCreator()
    rows = rtc.build_rows(files=files,
                          problem_header=problem_header,
                          result_header=result_header,
                          problems=problems)
    utils.toFile(rows, "result_solved_ipopt.html", date=False)


def table_solved_master():
    names = ['DECONVB', 'EIGENALS', 'FBRAIN3LS', 'OSCIPATH', 'PALMER5E', 'SSI']
    pyc = pycutest.Pycutest()
    problems = pyc.getProblems(names, aslist=True)
    problems = [p.problem_name for p in problems]
    rtc = RTCreator()
    rows = rtc.build_rows(files=files,
                          problem_header=problem_header,
                          result_header=result_header,
                          problems=problems)
    utils.toFile(rows, "result_solved_master.html", date=False)

# table_unsolved_ipopt_master()
# table_solved_ipopt()
# table_solved_master()

pyc = pycutest.Pycutest()
problems = pyc.getProblems(constraint=False,
                           # bounded=True,
                           aslist=True)
# problems = sorted(problems, key=lambda p: p.num_var)
problems = [p.problem_name for p in problems if p.num_var <= 5000]
# problems = problems[20:100]
rtc = RTCreator()
rows = rtc.build_rows(files=files,
                      problem_header=problem_header,
                      result_header=result_header,
                      problems=problems,
                      out="latex")
# utils.toFile(rows, "result.html", date=True)
