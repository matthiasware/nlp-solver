import numpy as np
from solver import Solver
from solver import NLP
import optiview as ov


class Rosenbrock(NLP):
    def __init__(self):
        # self.x0 = np.array([-1., 0])
        self.x0 = np.array([-1., -1.25])
        # self.lb = None
        # self.ub = None
        self.lb = np.array([-1.2, -1.5])
        self.ub = np.array([0.5, 0.5])
        self.plot_ranges = ((self.lb[0] - 0.25, self.ub[0] + 0.25),
                            (self.lb[1] - 0.25, self.ub[1] + 0.25))

    def fg(self, x):
        f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        g = np.array([400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2,
                      200 * (x[1] - x[0]**2)])
        return f, g

    def h(self, x):
        return np.array([[1200 * x[0]**2 - 400 * x[1] + 2,
                          -400 * x[0]],
                         [-400 * x[0],
                          200]])

solver = Solver(plot_steps=True)
result = solver.solve(Rosenbrock())


r = Rosenbrock()
v = ov.Visualizer()
v.plot(f=r.fg, history=solver.history, bounds=(r.lb, r.ub),
       ranges=((r.lb[0] - 0.25, r.ub[0] + 0.25), (r.lb[1] - 0.25 , r.ub[1] + 0.25)))
# v.plot(f=r.fg, history=solver.history)