import optiview
import pycutest
from solver import Solver
import sys
import traceback
from solver import NLP

v = optiview.Visualizer()
pyc = pycutest.Pycutest()
problems = pyc.getProblems(constraint=False,
                           bounded=True,
                           aslist=True)

# 3PK Reached maximum: 10001 of iterations
if False:
    problem = [p for p in problems if p.problem_name == "3PK"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u)
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)
# PFIT2LS
if False:
    problem = [p for p in problems if p.problem_name == "PFIT2LS"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u)
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)

# PFIT3LS
# maxiterations
if False:
    problem = [p for p in problems if p.problem_name == "PFIT3LS"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u)
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)

# PFIT4LS
# raise LinAlgError("Eigenvalues did not converge")
if False:
    problem = [p for p in problems if p.problem_name == "PFIT4LS"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u)
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)

# BOX2
if False:
    problem = [p for p in problems if p.problem_name == "BOX2"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u)
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)

# KOEBHELB
# - quadratic approximation leads to xbar which is not in feasible region
# - direction dx, linesearch leads to extremely high fv
# 113.36793657804775
# 95372932.72442278
# 114.17234402526644
# 112.22426594981489
# 112.22425395474379
if False:
    problem = [p for p in problems if p.problem_name == "KOEBHELB"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u)
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)


# SIM2BQP
if False:
    problem = [p for p in problems if p.problem_name == "SIM2BQP"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u,
              plot_ranges=((-1, 1, 100),
                           (-0.5, 1, 100)))
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)

    v.plot(nlp.fg, history=s.history, ranges=nlp.plot_ranges,
           bounds=(nlp.lb, nlp.ub))

# HIMMELP1
# STEPSIZE PROBLEM ?
if False:
    problem = [p for p in problems if p.problem_name == "HIMMELP1"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u,
              plot_ranges=((-1, 96, 100),
                           (-1, 76, 100)))
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)

    v.plot(nlp.fg, history=s.history, ranges=nlp.plot_ranges,
           bounds=(nlp.lb, nlp.ub))

# HS3
if True:
    problem = [p for p in problems if p.problem_name == "HS3"][0]
    nlp = NLP(fg=problem.fg, x0=problem.var_init, h=problem.h,
              lb=problem.var_bounds_l, ub=problem.var_bounds_u,
              plot_ranges=((-3, 15, 100),
                           (-3, 5, 100)))
    s = Solver()
    try:
        res = s.solve(nlp)
    except Exception:
        traceback.print_exc(file=sys.stdout)

    v.plot(nlp.fg, history=s.history, ranges=nlp.plot_ranges,
           bounds=(nlp.lb, nlp.ub))
