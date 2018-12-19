import numpy as np

# TODO
# overflow in min(min(a_star, (lb[i] - xc[i]) / d[di]))

def minsubspace(x, xc, lb, ub, g, H, f):
    n = len(xc)
    Z = []
    F = []
    for i in range(n):
        if xc[i] != lb[i] and xc[i] != ub[i]:
            Z.append(np.eye(1, n, i))
            F.append(i)
    if len(F) < 1:  # xc is already optimal
        return xc

    # Z = np.array(Z)
    # Z = Z.reshape((n, len(F)))
    Z = np.concatenate(Z).T

    rc = (Z.T).dot(g + H.dot(xc - x))
    Hc = (Z.T).dot(H).dot(Z)

    # search direction
    d = np.linalg.solve(Hc, -rc)

    a_star = 1  # max
    for di, i in enumerate(F):
        if d[di] > 0:
            a_star = min(a_star, (ub[i] - xc[i]) / d[di])
        elif d[di] < 0:
            a_star = min(a_star, (lb[i] - xc[i]) / d[di])

    # project optimum
    x_bar = np.copy(xc)  # (5.9)
    for di, i in enumerate(F):
        x_bar[i] = xc[i] + a_star * d[di]
        x_bar[i] = max(lb[i], min(ub[i], x_bar[i]))
    # assert np.all(lb <= x_bar) and np.all(x_bar <= ub), "subspace out of bounds"
    return x_bar
