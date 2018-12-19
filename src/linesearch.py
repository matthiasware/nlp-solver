def linesearch(x, dx, alpha_max, f, fk, gk):
    alpha = min(1, alpha_max)
    c = 1e-4
    p = 0.5
    while True:
        fv = f(x + alpha * dx)
        if fv <= fk + c * alpha * gk.dot(dx):
            break
        alpha = p * alpha
    return alpha


def wolfe(x, dx, max_iter, f, g):
    alpha = 0
    beta = 1000
    step = 5
    c1 = 0.15
    c2 = 0.3
    i = 0
    while i <= max_iter:
        leftf = f(x + step * dx)
        rightf = f(x) + c1 * alpha * g(x).dot(dx)
        if leftf > rightf:
            beta = step
            step = .5 * (alpha + beta)
        elif f(x + step * dx).dot(dx) < c2 * f(x).dot(dx):
            alpha = step
            if beta > 100:
                step = 2 * alpha
            else:
                step = .5 * (alpha + beta)
        else:
            break
        i += 1
    return step
