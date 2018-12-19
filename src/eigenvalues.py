import numpy as np
import scipy.sparse.linalg as sla
import timeit
from scipy.optimize import minimize
from matplotlib import pyplot as plt


np.random.seed(100)


def createHessian(n):
    H = np.random.rand(n, n)
    H = (H + H.T) / 2
    H = H + 10 * np.eye(n)
    return H


def ev_sp():
    v = sla.eigsh(H, k=1, which="SA", return_eigenvectors=False, tol=0)
    return np.real(v)


def ev_np():
    u, _ = np.linalg.eigh(H)
    return u[0]


def fAndG(x, A):
    t_0 = np.dot(x, x)
    t_1 = np.dot(A, x)
    t_2 = np.dot(x, t_1)
    f_ = (t_2 / t_0)
    g_0 = (((2 * (1 / t_0)) * t_1) - (2 * (((1 / (t_0 ** 2)) * t_2) * x)))
    g_ = g_0
    return f_, g_


def ev_soeren():
    n, _ = H.shape
    x0 = np.random.randn(n)
    result = minimize(fAndG, x0, args=H, jac=True, method='L-BFGS-B',
                      tol=1E-10)
    f = result.fun
    return f


if __name__ == "__main__":
    s_input = list(range(2, 10, 1))
    s_input += list(range(10, 100, 10))
    s_input += list(range(100, 1000, 50))
    times = 10
    sp_times = []
    np_times = []
    sr_times = []
    print("{:>4}\t{:>10}\t{:>10}\t{:>10}".format(
        "n", "sparse", "dense", "soeren"))
    for n in s_input:
        H = createHessian(n)
        sp_t = timeit.timeit("ev_sp()", number=times,
                             setup="from __main__ import ev_sp")
        np_t = timeit.timeit("ev_np()", number=times,
                             setup="from __main__ import ev_np")
        sr_t = timeit.timeit("ev_soeren()", number=times,
                             setup="from __main__ import ev_soeren")
        print("{:4d}\t{:>04.6f}\t{:>04.6f}\t{:>04.6f}".format(n, sp_t, np_t, sr_t))
        sp_times.append(sp_t)
        np_times.append(np_t)
        sr_times.append(sr_t)


plt.plot(s_input, sp_times, label="sparse")
plt.plot(s_input, np_times, label="dense")
plt.plot(s_input, sr_times, label="soere")
plt.xlabel("size: n of n x n Matrix")
plt.ylabel("runtime / seconds")
plt.legend()
plt.show()
