import numpy as np
import scipy.sparse.linalg as sla
from optimizationresult import OptimizationResult
from cauchy import cauchy
from cauchy import create_mx
from subspace import minsubspace
from plotter import Plot
import warnings
import myLineSearch as linesearch
import timeout_decorator
import hessian_modification as hm
import traceback
import itertools
import time
from scipy.optimize import minimize

warnings.filterwarnings('error')
MAXTIME = 360


def fAndG(x, A):
    t_0 = np.dot(x, x)
    t_1 = np.dot(A, x)
    t_2 = np.dot(x, t_1)
    f_ = (t_2 / t_0)
    g_0 = (((2 * (1 / t_0)) * t_1) - (2 * (((1 / (t_0 ** 2)) * t_2) * x)))
    g_ = g_0
    return f_, g_


def ev_soeren(H):
    n, _ = H.shape
    x0 = np.random.randn(n)
    result = minimize(fAndG, x0, args=H, jac=True, method='L-BFGS-B',
                      tol=1E-10)
    f = result.fun
    return f


class Solver:
    def __init__(self, ftol=1e-8, gtol=1e-8, verbose=True, maxiter=5000,
                 record=True, plot_steps=True, ls="wolfe2", hm="cur"):
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose
        self.maxiter = maxiter
        self.record = record
        self.plot_steps = plot_steps
        self.ls = ls
        self.hm = hm

    def solve(self, nlp):
        self.nlp = nlp
        self.history = []
        self.success = False
        self.message = None
        self.x = None
        self.g = None
        self.H = None
        self.f = None
        self.i = None
        t0 = time.perf_counter()
        try:
            if nlp.is_bounded:
                self._minbounded()
            else:
                self._minunbounded()
        except timeout_decorator.timeout_decorator.TimeoutError:
            self.message = "Timeout after {} sec.".format(MAXTIME)
        except AssertionError as e:
            self.message = e.args[0]
        except Exception as e:
            tb = traceback.format_exc()
            self.message = str(e) + "\n" + tb
        t1 = time.perf_counter()
        elapsed = t1 - t0
        res = OptimizationResult(x=self.x, success=self.success,
                                 status=None, message=self.message,
                                 fun=self.f, jac=self.g, hess=self.H,
                                 nit=self.i, nlp=self.nlp, tsec=elapsed)
        return res

    @timeout_decorator.timeout(MAXTIME)
    def _minbounded(self):
        # move x inside feasible region
        x = np.maximum(self.nlp.lb,
                       np.minimum(self.nlp.x0,
                                  self.nlp.ub))
        f_old = np.inf
        for i in itertools.count():
            f, g = self.nlp.fg(x)
            H = self._modify_hessian(self.nlp.h(x))
            self._set_variables(x, f, g, H, i)
            assert f <= f_old, "f >= fold"

            if not (self._cftol(f, f_old) and
                    self._cpgtol(x, g) and
                    self._cmaxiter(i)):
                break

            xc = cauchy(x, f, g, H, self.nlp.lb, self.nlp.ub)
            xs = minsubspace(x, xc, self.nlp.lb, self.nlp.ub, g, H, f)
            dx = xs - x
            assert g.dot(dx) < 0, "g.dot(dx) < 0"
            alpha_max = self._calc_alpha_max(x, dx)
            alpha = self._linesearch(x, dx, 20, alpha_max)
            x = x + alpha * dx
            x = np.maximum(self.nlp.lb,
                           np.minimum(x, self.nlp.ub))  # adjust numerics
            f_old = f

    @timeout_decorator.timeout(MAXTIME)
    def _minunbounded(self):
        if self.verbose:
            print("%4s\t%10s\t%10s\t%10s" % ("iter", "funVal",
                                             "step length",
                                             "gnorm"))
        f_old = np.inf
        x = np.copy(self.nlp.x0)
        for i in itertools.count():
            f, g = self.nlp.fg(x)
            H = self._modify_hessian(self.nlp.h(x))
            self._set_variables(x, f, g, H, i)
            assert f < f_old + 1e-6, "f > fold: {} > {}".format(f, f_old)
            if not (self._cftol(f, f_old) and
                    self._cgtol(g) and
                    self._cmaxiter(i)):
                break
            f_old = f
            dx = np.linalg.solve(H, -g)
            while(1):
                f2, g2 = self.nlp.fg(x + dx)
                if np.linalg.norm(g2, np.inf) < 1E10:
                    break
                dx /= 2.
            assert g.dot(dx) < 0, "g.dot(dx) < 0"
            alpha = self._linesearch(x, dx, 20)
            if self.verbose:
                print("%4i\t%10.6g\t%10.5g\t%10.5g"
                      % (i, f, alpha, np.linalg.norm(g, np.inf)))
            x = x + alpha * dx

    def _linesearch(self, x, dx, maxiter=20, alpha_max=None):
        alpha, *_ = linesearch.line_search_wolfe2(f=self.nlp.f,
                                                  myfprime=self.nlp.g,
                                                  xk=x,
                                                  pk=dx,
                                                  maxiter=maxiter,
                                                  amax=alpha_max)
        assert alpha is not None, "alpha is None"
        assert alpha > 0, "alpha <= 0"
        if alpha_max is not None:
            assert alpha <= alpha_max, "alpha > alpha_max"
        return alpha

    def _set_variables(self, x, f, g, H, i):
        self.x = np.copy(x)
        self.f = np.copy(f)
        self.g = np.copy(g)
        self.H = np.copy(H)
        self.i = i
        if self.record:
            self.history.append(np.copy(x))

    def _calc_alpha_max(self, x, dx):
        alpha_max = np.inf
        for j in range(len(x)):
            if dx[j] > 0:
                alpha_max = min(alpha_max, (self.nlp.ub[j] - x[j]) / dx[j])
            if dx[j] < 0:
                alpha_max = min(alpha_max, (self.nlp.lb[j] - x[j]) / dx[j])
        return alpha_max

    def _cftol(self, f, f_old):
        if np.abs(f_old - f) < self.ftol:
            self.message = "|f_old - f| < ftol"
            self.success = True
            return False
        return True

    def _cpgtol(self, x, g):
        g_projected = self._projected(x - g) - x
        if np.linalg.norm(g_projected, np.inf) < self.gtol:
            self.message = "|g_p| < gtol"
            self.success = True
            return False
        return True

    def _projected(self, x):
        for i in range(len(x)):
            if x[i] < self.nlp.lb[i]:
                x[i] = self.nlp.lb[i]
            elif x[i] > self.nlp.ub[i]:
                x[i] = self.nlp.ub[i]
        return x

    def _cmaxiter(self, i):
        if i > self.maxiter:
            self.message = "Reached maximum: %i of iterations" % i
            self.success = False
            return False
        return True

    def _cgtol(self, g):
        if np.linalg.norm(g, np.inf) < self.gtol:
            self.message = "|g_p| < gtol"
            self.success = True
            return False
        return True

    def _modify_hessian(self, H):
        H = np.copy(H)
        if len(self.nlp.x0) < 600:
            w, _ = np.linalg.eigh(H)
            w = w[0]
        else:
            w = ev_soeren(H)
        if w < 1e-5:
            H.flat[::H.shape[0] + 1] += (- w + 1)  # 1E-6 * np.abs(w[-1]))
        return H
