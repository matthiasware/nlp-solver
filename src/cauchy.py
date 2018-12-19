import numpy as np


def create_mx(f, g, H, x0):
    def mx(x, x0=x0, f=f, g=g, H=H):
        return f + g.dot(x - x0) + 0.5 * (x - x0).dot(H).dot(x - x0)
    return mx


def create_mt(fv, fp, fpp):
    def mt(dt):
        return fv + fp * dt + 0.5 * fpp * dt**2
    return mt


def init_descent_direction(t, g):
    n = len(g)
    d = np.zeros(n)
    for i in range(n):
        # calculate descent direction
        # if we are already on bounds, we
        # dont go further in this direction
        if t[i] == 0:
            d[i] = 0
        # otherwise we use the direction of -g
        else:  # t[i] > 0:
            d[i] = - g[i]
    return d


def init_free_indices(t):
    F = [i for i, ti in enumerate(t) if ti > 0]
    return F


def calculate_f(f, g, d, z, H):
    fv = f + g.dot(z) + 0.5 * z.dot(H).dot(z)
    fvp = g.dot(d) + d.dot(H).dot(z)
    fvpp = d.dot(H).dot(d)
    return fv, fvp, fvpp


def init_breakpoints(x, lb, ub, g):
    n = len(x)
    t = np.zeros(n)
    for i in range(n):
        if g[i] < 0:
            t[i] = (x[i] - ub[i]) / g[i]
        elif g[i] > 0:
            t[i] = (x[i] - lb[i]) / g[i]
        else:
            t[i] = np.inf
    return t


def cauchy(x0, f, g, H, lb, ub):
    ts = init_breakpoints(x0, lb, ub, g)
    F = init_free_indices(ts)
    d = init_descent_direction(ts, g)  # d_(j-1)
    # b = b[i] is i-th smallest t = ts[b]
    bs = [b for b in np.argsort(ts) if b in F]
    xc = np.zeros(len(x0))
    xc[:] = x0
    z = xc - x0  # z_(j-1)
    fp = g.dot(d)  # (4.4) fp_(j-1)
    fpp = d.dot(H).dot(d)  # (4.5) fpp_(j-1)

    dt_min = - fp / fpp
    t_old = 0
    for b in bs:
        t = ts[b]  # t_(j-1)
        dt = t - t_old  # dt_(j-1)
        if dt_min <= dt:
            break
        t_old = t
        F.remove(b)
        if d[b] > 0:
            xc[b] = ub[b]
        elif d[b] < 0:
            xc[b] = lb[b]  # xc_j
        d[b] = 0  # (4.7)
        z = z + dt * d  # (4.8) z_j
        fp = g.dot(d) + d.dot(H).dot(z)  # (4.9) fp_j
        fpp = d.dot(H).dot(d)  # (4.10) fpp_j
        if fp == 0 or fpp == 0:
            dt_min = 0
        else:
            dt_min = - fp / fpp
    dt_min = max(dt_min, 0)
    t_old = t_old + dt_min
    for i in F:
        xc[i] = x0[i] + d[i] * t_old
    # assert not np.allclose(x0, xc, rtol=1e-10), "x0 alomost equal to xc"
    assert np.all(lb <= xc) and np.all(xc <= ub), "xc out of bounds"
    return xc
