import matplotlib.pyplot as plt
import numpy as np
from functions import *


class Plot2():

    def __init__(self, legend=False):
        self.actions = []
        self.max_ranges = None
        self.outer_range = 0.1
        self.legend = legend

    def updataMaxRanges(self, ranges):
        if self.max_ranges is None:
            self.max_ranges = ranges
        else:
            self.max_ranges[0][0] = min(self.max_ranges[0][0], ranges[0][0])
            self.max_ranges[1][0] = min(self.max_ranges[1][0], ranges[1][0])
            self.max_ranges[0][1] = max(self.max_ranges[0][1], ranges[0][1])
            self.max_ranges[1][1] = max(self.max_ranges[1][1], ranges[1][1])

    def clean(self):
        self.actions = []

    def undo(self):
        self.actions.pop()

    def redo(self):
        pass

    def scatter(self, x, color=None, alpha=1,
                size=40, title=None, name=None):
        x, y = zip(*x)

        def action(x=x, y=y, color=color, alpha=alpha,
                   size=size):
            plt.scatter(x, y, alpha=alpha, s=size, c=color)
        self.actions.append(action)

    def dline(self, x0, d, lb=None, ub=None, color="yellow"):
        # directional line along x0 + td
        if lb is None:
            lb = [self.max_ranges[0][0], self.max_ranges[1][0]]
        if ub is None:
            ub = [self.max_ranges[0][1], self.max_ranges[1][1]]
        tl = [(lb[0] - x0[0]) / d[0], (lb[1] - x0[1]) / d[1]]
        il = np.argmin(np.abs(tl))
        tu = [(ub[0] - x0[0]) / d[0], (ub[1] - x0[1]) / d[1]]
        iu = np.argmin(np.abs(tu))

        def action(x0=x0, tl=tl, ub=ub, il=il, iu=iu, d=d, color=color):
            points = [x0 + tl[il] * d, x0 + tu[iu] * d]
            x, y = zip(*points)
            plt.plot(x, y, color=color)
        self.actions.append(action)

    def pline(self, x, color=None, points=True):  # picewise line
        x, y = zip(*x)

        def action(x=x, y=y, points=points, color=color):
            plt.plot(x, y, color=color)
            if points:
                plt.scatter(x[1:-1], y[1:-1], c="b")
                plt.scatter(x[0], y[0], c="green")
                plt.scatter(x[-1], y[-1], c="red")
        self.actions.append(action)

    def vlines(self, x, color="magenta", linestyle="-"):
        for i in x:
            def action(i=i):  # !!! closure !!!
                plt.axvline(x=i, color=color, linestyle=linestyle)
            self.actions.append(action)

    def hlines(self, x, color="magenta", linestyle="-"):
        for i in x:
            def action(i=i):  # !!! closure !!!
                plt.axhline(y=i, color=color, linestyle=linestyle)
            self.actions.append(action)

    def contour(self, f=None, grad=False,
                ranges=None, colors="r", levels=20,
                fill_all=True):
        if ranges:
            self.updataMaxRanges(ranges)
            self.ranges = ranges
        if grad:
            def fnew(x):
                fv, _ = f(x)
                return fv
            self.f = fnew
        else:
            self.f = f
        x = np.linspace(*self.ranges[0])
        y = np.linspace(*self.ranges[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                x = np.array([X[i][j], Y[i][j]])
                Z[i, j] = self.f(np.array(x))
        if fill_all:
            def contour_action():
                cp = plt.contourf(X, Y, Z, levels)
                plt.colorbar(cp)
            self.actions.append(contour_action)
        else:
            def contour_action():
                plt.contour(X, Y, Z, levels, colors=colors)
            self.actions.append(contour_action)

    def show(self):
        plt.figure(figsize=(15, 15))
        for action in self.actions:
            action()
            # action[0](*action[1])
        plt.show()

    def setRangesFromPoints(self, history):
        xs, ys = zip(*history)
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        diff_x = np.abs(x_max - x_min)
        diff_y = np.abs(y_max - y_min)
        x_min = x_min - diff_x * self.outer_range
        y_min = y_min - diff_y * self.outer_range
        x_max = x_max + diff_x * self.outer_range
        y_max = y_max + diff_y * self.outer_range
        self.ranges = ((x_min, x_max, 100),
                       (y_min, y_max, 100))


class Plot():

    def __init__(self, legend=True):
        self.actions = []
        self.ranges = None
        self.outer_range = 0.1
        self.legend = legend

    def updateRanges(self, ranges):
        if self.ranges is None:
            self.ranges = ranges
        else:
            r = [[0, 0], [0, 0]]
            r[0][0] = min(self.ranges[0][0], ranges[0][0])
            r[1][0] = min(self.ranges[1][0], ranges[1][0])
            r[0][1] = max(self.ranges[0][1], ranges[0][1])
            r[1][1] = max(self.ranges[1][1], ranges[1][1])
            self.ranges = r

    def reset(self):
        self.actions = []

    def undo(self):
        self.actions.pop()

    def scatter(self, points, color=None, alpha=None,
                size=40, name=None,
                marker=None, label=None):
        x, y = zip(*points)

        def action(x=x, y=y, color=color, alpha=alpha,
                   size=size, label=label, marker=marker):
            plt.scatter(x, y, alpha=alpha, s=size, c=color,
                        label=label, marker=marker)
        self.actions.append(action)

    def dline(self, x0, d, fromx=False, tmax=None,
              lb=None, ub=None, color="yellow", label=None):
        # directional line along x0 + td
        if lb is None:
            lb = [self.ranges[0][0], self.ranges[1][0]]
        if ub is None:
            ub = [self.ranges[0][1], self.ranges[1][1]]
        # calculate breakpoints
        t0 = -np.inf
        t1 = np.inf
        for i in range(len(d)):
            if d[i] == 0:
                continue
            t = (lb[i] - x0[i]) / d[i]
            if t < 0:
                t0 = max(t0, t)
            elif t > 0:
                t1 = min(t1, t)
            else:
                t0 = 0
            t = (ub[i] - x0[i]) / d[i]
            if t < 0:
                t0 = max(t0, t)
            elif t > 0:
                t1 = min(t1, t)
            else:
                t1 = 0
        if tmax:
            t1 = min(t1, tmax)

        def action(x0=x0, t0=t0, t1=t1, d=d, color=color,
                   fromx=fromx, label=label):
            if fromx:
                points = [x0, x0 + t1 * d]
            else:
                points = [x0 + t0 * d, x0 + t1 * d]
            x, y = zip(*points)
            plt.plot(x, y, color=color, label=label)
        self.actions.append(action)

    def pline(self, points, linestyle=None,
              color=None,
              plot_points=True,
              palpha=None, pmarker=None, psize=None,
              label=None):  # picewise line
        x, y = zip(*points)

        def action(x=x, y=y, linestyle=linestyle, color=color,
                   plot_points=plot_points, palpha=palpha,
                   pmarker=pmarker, psize=psize, label=label):
            plt.plot(x, y, color=color, label=label)
            if plot_points:
                plt.scatter(x, y, color=color, marker=pmarker,
                            s=psize)
        self.actions.append(action)

    def vlines(self, x, color="magenta", linestyle="-", label=None):
        for i in x:
            def action(i=i, color=color, linestyle=linestyle, label=label):
                plt.axvline(x=i, color=color, linestyle=linestyle, label=label)
            self.actions.append(action)

    def hlines(self, x, color="magenta", linestyle="-", label=None):
        for i in x:
            def action(i=i, color=color, linestyle=linestyle, label=label):
                plt.axhline(y=i, color=color, linestyle=linestyle, label=label)
            self.actions.append(action)

    def contour(self, f=None, grad=False, ranges=None,
                colors="r", levels=40, fill_all=True):
        if ranges is None:
            ranges = self.ranges
        else:
            self.updateRanges(ranges)
        if grad:
            def fnew(x):
                fv, _ = f(x)
                return fv
            self.f = fnew
        else:
            self.f = f
        x = np.linspace(*ranges[0])
        y = np.linspace(*ranges[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                x = np.array([X[i][j], Y[i][j]])
                Z[i, j] = self.f(np.array(x))
        if fill_all:
            def contour_action(X=X, Y=Y, Z=Z, levels=levels):
                cp = plt.contourf(X, Y, Z, levels)
                plt.colorbar(cp)
            self.actions.append(contour_action)
        else:
            def contour_action(X=X, Y=Y, Z=Z, levels=levels, colors=colors):
                plt.contour(X, Y, Z, levels, colors=colors)
            self.actions.append(contour_action)

    def show(self, figsize=(15, 15)):
        plt.figure(figsize=figsize)
        for action in self.actions:
            action()
            # action[0](*action[1])
        # fig.legend(["blabla", "bla2"])
        if self.legend:
            plt.legend()
        plt.show()

    def setRangesFromPoints(self, history):
        xs, ys = zip(*history)
        xs = [x for x in xs if np.abs(x) != np.inf]
        ys = [y for y in ys if np.abs(y) != np.inf]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        diff_x = np.abs(x_max - x_min)
        diff_y = np.abs(y_max - y_min)
        x_min = x_min - diff_x * self.outer_range
        y_min = y_min - diff_y * self.outer_range
        x_max = x_max + diff_x * self.outer_range
        y_max = y_max + diff_y * self.outer_range
        self.ranges = ((x_min, x_max, 100),
                       (y_min, y_max, 100))


if False:
    r = rosenbrock()
    plot_ranges = [[-5, 5, 100], [-5, 5, 100]]
    lb = [-4, -1]
    ub = [3, 2]
    x = np.array([1, 1])
    d = np.array([-1, -1])
    plot = Plot()
    plot.contour(f=r.fg, grad=True, ranges=plot_ranges, fill_all=False)
    plot.hlines([lb[1], ub[1]])
    plot.vlines([lb[0], ub[0]])
    plot.scatter([x])
    # plot.dline(x, d, lb=lb, ub=ub, fromx=False)
    plot.dline(x, d, fromx=True, tmax=1, color="green", label="Gradient")
    plot.show()

if False:
    r = rosenbrock()
    x0 = r.x0

    f, g = r.fg(x0)
    H = r.h(x0)

    d = -g
    fp = g.dot(d)
    fpp = d.dot(H).dot(d)
    dt_min = - fp / fpp

    def m(x):
        return f + g.dot(x - x0) + 0.5 * (x - x0).dot(H).dot(x - x0)

    plot = Plot2()
    plot.outer_range = 0
    plot.setRangesFromPoints([r.lb, r.ub, x0])
    plot.contour(f=r.fg, grad=True)
    plot.contour(f=r.fg, grad=True, ranges=r.plot_ranges, fill_all=False)
    plot.contour(f=m, grad=False, fill_all=False, colors="blue")
    plot.dline(x0, d, color="green")
    # plot.pline([x0, x0 + dt_min * d, x0 + dt_min * 2 * d],
               # label="Bla")
    # plot.scatter([x0 * 2, x0 * 2 - dt_min * g], color="red")
    # plot.vlines([r.lb[0], r.ub[0]], label="bounds")
    # plot.hlines([r.lb[1], r.ub[1]])
    # plot.scatter([x0, x0 + dt_min * d], label="some points", marker="o")
    plot.show()
