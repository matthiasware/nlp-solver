import numpy as np
import optiview as ov
from plotter import Plot
from solver import Solver
from nlp import NLP


def eig(H):
    u, U = np.linalg.eig(H)
    return u


def inv(H):
    return np.linalg.inv(H)


def create_mx(f, g, H, x0):
    def mx(x, x0=x0, f=f, g=g, H=H):
        return f + g.dot(x - x0) + 0.5 * (x - x0).dot(H).dot(x - x0)
    return mx


def f(x):
    return 1 / 6 * x[0]**3 + 1 / 6 * x[1]**3


def g(x):
    return 1 / 3 * np.array([x[0]**2, x[1]**2])


def fg(x):
    return f(x), g(x)


def H(x):
    return np.array([[x[0], 0],
                     [0, x[1]]])


lb = np.array([-3, -3])
ub = np.array([3, 3])

v = ov.Visualizer()
ranges = ((-4, 4, 100), (-4, 4, 100))

x = np.array([-1., 1])  # Indefinite
# x = np.array([-1, -1])
x = np.array([-1, -1])  # negative definite
fv = f(x)
gv = g(x)
Hv = H(x)
dx = np.linalg.solve(Hv, -gv)

m = create_mx(fv, gv, Hv, x)

nlp = NLP(fg=fg, h=H, x0=x, lb=lb, ub=ub)
s = Solver()
res = s.solve(nlp)

if True:
    p = Plot()
    p.contour(f=f, grad=False, ranges=ranges)
    p.contour(f=m, grad=False, fill_all=False)
    p.vlines(np.array([-3, 3]))
    p.hlines(np.array([-3, 3]))
    p.scatter([x], color="green")
    p.dline(x, dx, color="black")
    p.show()

if True:
    v.plot(m, grad=False, ranges=ranges, history=[x], d3=True)
    # v.plot(f, grad=False, ranges=ranges, history=s.history, d3=True)
