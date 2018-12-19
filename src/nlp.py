import numpy as np


class NLP:
    def __init__(self, fg, x0, h, lb=None, ub=None, name=None,
                 plot_ranges=None):
        self.fg = fg
        self.x0 = x0
        self.h = h
        self.lb = lb
        self.ub = ub
        self.plot_ranges = plot_ranges
        self.name = name

    fg = None
    x0 = None
    lb = None
    ub = None
    name = None
    _is_bounded = None
    plot_ranges = None

    @property
    def is_bounded(self):
        if self._is_bounded is None:
            if self.lb is None and self.ub is None:
                self._is_bounded = False
            elif np.all(self.lb == -np.inf) and np.all(self.ub == np.inf):
                self._is_bounded = False
            else:
                self._is_bounded = True
        return self._is_bounded

    def f(self, x):
        fv, _ = self.fg(x)
        return fv

    def g(self, x):
        _, gv = self.fg(x)
        return gv
