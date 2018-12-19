"""
TODO:
- sören: result object should contain information why
         process failed or terminated succesfully
- scipy returns non ...
"""
import abc
import pycutest
from optimizationresult import OptimizationResult
from nlp import NLP
from scipy.optimize import minimize
import timeout_decorator
import time
import numpy as np
from subprocess import check_output
import os
from solver import Solver
import sys
import datetime
import csv
sys.path.append('/home/matthias/projects/genosolver/src/pygeno/')
import genosolver

MAXTIME = 360


class Driver(abc.ABC):
    @abc.abstractmethod
    def run(self, problem):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class ScipyDriver(Driver):
    def __init__(self, ftol=1e-8, gtol=1e-8, maxiter=5000):
        self.ftol = ftol
        self.gtol = gtol
        self.maxiter = maxiter

    @property
    def name(self):
        return "scipy.bfgs"

    def run(self, problem):
        result = OptimizationResult()
        try:
            t0 = time.perf_counter()
            res = self.runBFGS(problem)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            if np.isnan(res.fun):
                res.fun = None
            result = OptimizationResult(x=res.x,
                                        success=res.success,
                                        status=res.status,
                                        message=res.message,
                                        fun=res.fun,
                                        nit=res.nit,
                                        tsec=elapsed)
        except timeout_decorator.timeout_decorator.TimeoutError:
            result.success = False
            result.message = "Timeout after {} sec.".format(MAXTIME)

        return result

    @timeout_decorator.timeout(MAXTIME)
    def runBFGS(self, problem):
        if problem.is_bounded:
            res = minimize(fun=problem.fg,
                           jac=True,
                           x0=problem.var_init,
                           options={"maxiter": self.maxiter,
                                    "gtol": self.gtol,
                                    "disp": False},
                           method="L-BFGS-B",
                           bounds=list(zip(problem.var_bounds_l,
                                           problem.var_bounds_u)),
                           tol=self.ftol)
            res.message = res.message.decode('ascii')
        else:
            res = minimize(fun=problem.fg,
                           jac=True,
                           x0=problem.var_init,
                           method="BFGS",
                           options={"maxiter": self.maxiter,
                                    "gtol": self.gtol,
                                    "norm": np.inf,
                                    "disp": False},
                           tol=self.ftol)
        return res


class IpoptDriver(Driver):
    def __init__(self):
        pass

    @property
    def name(self):
        return "ipopt"

    @timeout_decorator.timeout(MAXTIME)
    def _run(self, problem):
        output = check_output(["runcutest", "-p", "ipopt",
                               "-D", problem.problem_name])
        return str(output)

    def run(self, problem):
        wd = os.getcwd()
        os.chdir("/tmp")
        try:
            t0 = time.perf_counter()
            output = self._run(problem)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            result = self._createOptimizationResult(output)
            result.tsec = elapsed
        except timeout_decorator.timeout_decorator.TimeoutError:
            result = OptimizationResult()
            result.success = False
            result.message = "Timeout after {} sec.".format(MAXTIME)
        os.chdir(wd)
        return result

    def _createOptimizationResult(self, output):
        output = output.split("\\n")
        status_code = output[-9].split("=")
        if not status_code[0].strip() == "Exit code":
            status_code = None
        else:
            status_code = int(status_code[-1])
        exit_msg = output[-25].split(":")
        if not exit_msg[0] == "EXIT":
            exit_msg = None
        else:
            exit_msg = exit_msg[1].strip()
        f_final = output[-8].split("=")
        if not f_final[0].strip() == "Final f":
            f_final = None
        else:
            try:
                f_final = float(f_final[-1].strip())
            except ValueError:
                f_final = None
        iterations = output[-45].split(":")
        if not iterations[0] == "Number of Iterations....":
            iterations = None
        else:
            iterations = int(iterations[-1])
        result = OptimizationResult()
        result.status = status_code
        result.message = exit_msg
        result.fun = f_final
        result.nit = iterations
        result.success = status_code == 0 or status_code == 1
        return result


class MasterDriver(Driver):
    def __init__(self, ftol=1e-8, gtol=1e-8, maxiter=5000):
        self.ftol = ftol
        self.gtol = gtol
        self.maxiter = maxiter

    @property
    def name(self):
        return "master"

    def run(self, problem):
        nlp = NLP(fg=problem.fg,
                  x0=problem.var_init,
                  h=problem.h,
                  lb=problem.var_bounds_l,
                  ub=problem.var_bounds_u)
        solver = Solver(ftol=self.ftol, gtol=self.gtol,
                        maxiter=self.maxiter, verbose=False,
                        record=False)
        res = solver.solve(nlp)
        res.H = None  # Memory issues ;)
        res.g = None
        return res


class GenoDriver(Driver):
    def __init__(self, ftol=1e-8, gtol=1e-8, maxiter=5000):
        self.ftol = ftol
        self.gtol = gtol
        self.maxiter = maxiter

    @property
    def name(self):
        return "geno"

    def run(self, problem):
        result = OptimizationResult()
        try:
            res = self._run(problem)
            result.success = res.success
            result.status = 0 if res.success else -1
            result.fun = res.fun
            result.nit = res.nit
            result.tsec = res.elapsed
            result.message = ""
        except timeout_decorator.timeout_decorator.TimeoutError:
            result.success = False
            result.message = "Timeout after {} sec.".format(MAXTIME)
        return result

    @timeout_decorator.timeout(MAXTIME)
    def _run(self, problem):
        bounds = None
        if problem.is_bounded:
            bounds = list(zip(problem.var_bounds_l,
                              problem.var_bounds_u))
        res = genosolver.minimize(
            fg=problem.fg,
            x0=problem.var_init,
            bounds=bounds,
            tol=self.ftol,
            options={"maxiter": self.maxiter})
        return res


class RunCutest:
    def __init__(self):
        self.header = ["name", "bounds", "n", "m",
                       "success", "status", "iter",
                       "f", "tsec", "msg"]

    def run(self, driver, problems=None):
        if problems is None:
            pyc = pycutest.Pycutest()
            problems = pyc.getProblems(aslist=True)
        results = []
        for i, problem in enumerate(problems):
            print("{:4g}/{:4g}\t{:12}\t".format(i, len(problems),
                                                problem.problem_name),
                  end="", flush=True)
            result = driver.run(problem)
            print(result.success)
            results.append((problem, result))
        rows = self._resultsToCsvRows(results)
        return rows

    def _resultsToCsvRows(self, results):
        header = ["name", "bounds", "n", "m",
                  "success", "status", "iter", "f", "tsec", "msg"]
        rows = [header]
        for p, r in results:
            rows.append([
                p.problem_name,
                p.is_bounded,
                p.num_var,
                p.num_const,
                r.success,
                r.status,
                r.nit,
                r.fun,
                r.tsec,
                r.message
            ])
        return rows

    def toCsv(self, rows, file, date=True, append=False):
        if date:
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            file = str(date) + "-" + file
        mode = "a" if append else "w"
        with open(file, mode) as file:
            writer = csv.writer(file)
            writer.writerows(rows)


def toCsv(rows, file, date=True, append=False):
    if date:
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        file = str(date) + "-" + file
    mode = "a" if append else "w"
    with open(file, mode) as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == "__main__":
    pyc = pycutest.Pycutest()
    problems = pyc.getProblems(aslist=True,
                               # bounded=True,
                               constraint=False)
    # problems = [i for i in problems if i.problem_name not in blacklist]
    # problems = problems[0:100]
    # names failed
    names = ['BDEXP', 'BIGGSB1', 'BQPGAUSS', 'BROYDN7D',
             'CHAINWOO', 'CHENHARK', 'DIXMAAND', 'DIXMAANI',
             'DIXMAANJ', 'DIXMAANK', 'DIXMAANL', 'DIXMAANM',
             'DIXMAANN', 'DIXMAANO', 'DIXMAANP', 'DQRTIC',
             'DRCAV1LQ', 'DRCAV2LQ', 'DRCAV3LQ', 'EIGENALS',
             'EIGENBLS', 'EIGENCLS', 'EXPQUAD', 'FLETBV3M',
             'FLETCBV2', 'FLETCBV3', 'FLETCHBV', 'GENHUMPS',
             'HARKERP2', 'INDEF', 'JIMACK', 'MCCORMCK', 'MOREBV',
             'MSQRTALS', 'MSQRTBLS', 'NCB20B', 'NONCVXU2',
             'NONCVXUN', 'NONDQUAR', 'NONMSQRT', 'PENALTY1',
             'PENTDI', 'POWELLBC', 'QRTQUAD', 'QUARTC', 'QUDLIN',
             'RAYBENDL', 'RAYBENDS', 'SBRYBND', 'SCOSINE',
             'SPARSINE', 'SSBRYBND', 'SSCOSINE', 'TESTQUAD',
             'WALL10', 'YATP1LS']
    # names timeout 120 sec
    names = ['FLETCHCR', 'HARKERP2', 'PENALTY1',
             'POWELLBC', 'QR3DLS', 'VARDIM']
    problems = [i for i in problems if i.problem_name in names]

    driver = ScipyDriver()
    driver = GenoDriver()
    driver = IpoptDriver()

    driver = MasterDriver()

    rc = RunCutest()
    csvrows = rc.run(driver, problems)
    rc.toCsv(csvrows,
             driver.name + "-soeren-" ".csv",
             date=True,
             append=False)
