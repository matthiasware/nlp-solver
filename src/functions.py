import numpy as np
import random


class rosenbrock:
    def __init__(self, bp="ul"):
        pass
    x0 = np.array([-1., 0])
    lb = np.array([-1.2, -1.5])
    ub = np.array([0.5, 0.5])
    plot_ranges = ((-1.3, 1.2, 100), (-1.7, 1.2, 100))

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


class quadratic2:
    plot_ranges = ((-10, 10, 100), (-10, 10, 100))
    minimum = np.array([0, 0])

    def __init__(self, a, b):
        self.x0 = np.array([1.8, 3.8])
        self.lb = np.array([1, 1])
        self.ub = np.array([4, 4])
        self.a = a
        self.b = b

    def fg(self, x):
        f = self.a * x[0]**2 + self.b * x[1]**2
        g = np.array([self.a * 2 * x[0],
                      self.b * 2 * x[1]])
        return f, g

    def h(self, x):
        return np.array([[self.a * 2, 0],
                         [0, self.b * 2]])

    def init_next_problem(self, rsize=2):
        lb = np.array([-5, 1])
        ub = np.array([-1, 5])
        for v in range(3):
            for h in range(3):
                t = np.array([3 * h, -3 * v])
                lbn = lb + t
                ubn = ub + t
                for x0 in [lbn, ubn]:
                    yield lbn, ubn, x0
                for xx in [lbn[0], ubn[0]]:
                    xy = random.uniform(lbn[1], ubn[1])
                    x0 = np.array([xx, xy])
                    yield lbn, ubn, x0
                for xy in [lbn[1], ubn[1]]:
                    xx = random.uniform(lbn[0], ubn[0])
                    x0 = np.array([xx, xy])
                    yield lbn, ubn, x0
                for g in range(rsize):
                    x0 = np.empty(2)
                    x0[0] = random.uniform(lbn[0], ubn[0])
                    x0[1] = random.uniform(lbn[1], ubn[1])
                    yield lbn, ubn, x0


class quadratic:

    plot_ranges = ((-10, 10, 100), (-10, 10, 100))
    minimum = np.array([0, 0])

    problems = ["ur", "ul", "ll", "lr",
                "c", "cur", "cul", "cll", "clr"]

    def __init__(self, bp="ul", x0=None):
        self.plot_ranges = ((-5, 5, 100), (-5, 5, 100))
        if bp == "ur":
            self.x0 = np.array([1.8, 3.8])
            self.lb = np.array([1, 1])
            self.ub = np.array([4, 4])
        if bp == "ul":
            self.x0 = np.array([-1.8, 3.8])
            self.lb = np.array([-4, 1])
            self.ub = np.array([-1, 4])
        if bp == "ll":
            self.x0 = np.array([-1.8, -3.8])
            self.lb = np.array([-4, -4])
            self.ub = np.array([-1, -1])
        if bp == "lr":
            self.x0 = np.array([1.8, -3.8])
            self.lb = np.array([1, -4])
            self.ub = np.array([4, -1])
        if bp == "c":
            self.x0 = np.array([1.8, 3.8])
            self.lb = np.array([-1, -1])
            self.ub = np.array([4, 4])
        if bp == "cur":
            self.x0 = np.array([2, 3.8])
            self.lb = np.array([-1, 1])
            self.ub = np.array([4, 4])
        if bp == "cul":
            self.x0 = np.array([-2, 3.8])
            self.lb = np.array([-4, -1])
            self.ub = np.array([1, 4])
        if bp == "cll":
            self.x0 = np.array([-2, -3.8])
            self.lb = np.array([-4, -4])
            self.ub = np.array([1, -1])
        if bp == "clr":
            self.x0 = np.array([2, -3.8])
            self.lb = np.array([1, -4])
            self.ub = np.array([4, 1])
        if x0 is not None:
            self.x0 = np.copy(x0)

    def fg(self, x):
        f = x.dot(x)
        g = 2 * x
        return f, g

    def h(self, x):
        return np.array([[2, 0], [0, 2]])

    def getCauchy(self):
        m = self.minimum
        lb = self.lb
        ub = self.ub
        xc = np.copy(m)
        for i in range(len(xc)):
            if ub[i] < m[i]:
                xc[i] = ub[i]
            elif lb[i] > m[i]:
                xc[i] = lb[i]
            else:
                xc[i] = m[i]
        return xc
