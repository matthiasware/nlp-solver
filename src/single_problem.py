import optiview as ov
import pycutest
from solver import Solver
from nlp import NLP

# name = "BLEACHNG"  # f >= fold
# name = "HIMMELBF"  # maxiterations
# name = "HAHN1LS"  # f >= fold
name = "CHENHARK"

vr = ov.Visualizer()
pyc = pycutest.Pycutest()

p = pyc.getProblems(name)[name]

nlp = NLP(fg=p.fg, x0=p.var_init, h=p.h,
          lb=p.var_bounds_l, ub=p.var_bounds_u)

solver = Solver(ftol=1E-8, gtol=1E-8, verbose=False)
res = solver.solve(nlp)
print(res)

if p.num_var == 2:
    vr.plot(f=p.fg,
            history=solver.history,
            orange=.5,
            title=p.problem_name,
            d3=False)
