import optiview
from functions import *
from cauchy import *

v = optiview.Visualizer()

if True:
    problem = rosenbrock()
    x0 = problem.x0
    lb = problem.lb
    ub = problem.ub
    f, g = problem.fg(x0)
    H = problem.h(x0)

    xc = cauchy(x0, f, g, H, lb, ub)

    v.plot(f=problem.fg, ranges=problem.plot_ranges,
           d3=False,
           bounds=[problem.lb, problem.ub],
           history=[x0, xc],
           levels=40)

if True:
    for p in quadratic.problems:
        problem = quadratic(bp=p)
        x0 = problem.x0
        lb = problem.lb
        ub = problem.ub

        x = x0
        f, g = problem.fg(x)
        H = problem.h(x)

        xc = cauchy(x, f, g, H, lb, ub)
        v = optiview.Visualizer()
        v.plot(f=problem.fg, ranges=problem.plot_ranges,
               d3=False,
               bounds=[problem.lb, problem.ub],
               history=[x0, xc],
               levels=20)

if True:
    problem = quadratic2(a=10, b=1)
    generator = problem.init_next_problem(rsize=4)
    for lb, ub, x0 in generator:
        f, g = problem.fg(x0)
        H = problem.h(x0)
        print("*" * 16)
        xc = cauchy(x0, f, g, H, lb, ub)
        print("lb", lb)
        print("lb", ub)
        print("x0", x0)
        print("xc", xc)
        v.plot(f=problem.fg, ranges=problem.plot_ranges,
               d3=False,
               bounds=[lb, ub],
               history=[x0, xc],
               levels=40)
