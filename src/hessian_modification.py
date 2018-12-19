import numpy as np
import scipy.linalg as la
# set direction to 0
#

g = np.array([-1, -1, -1])
H = np.array([[0, 1, 1],
              [1, 1, -2],
              [1, -2, -3]], dtype=np.double)

# u, V = la.eig(H)
u, U = np.linalg.eig(H)
w, W = np.linalg.eigh(H)

H_u = U.dot(np.diag(u).dot(la.inv(U)))
H_w = W.dot(np.diag(w).dot(la.inv(W)))

# flip sign of negative ev
# set negative ev to 0


def h_spectral_cur(H, e=1):
    H = np.copy(H)
    w, _ = np.linalg.eigh(H)
    if w[0] < 1e-5:
        H.flat[::H.shape[0] + 1] += (- w[0] + e)  # 1E-6 * np.abs(w[-1]))
    return H


# same as eh_spectral_cur
def h_spectral_cc(H, delta=1, omege=0):
    w, W = np.linalg.eig(H)
    t = max(omege, delta - min(w))
    return H + np.eye(len(w)) * t


def h_spectral_flip(H):
    w, W = np.linalg.eig(H)
    w = np.abs(w)
    return W.dot(np.diag(w).dot(la.inv(W)))


# zero out, not necessarily descent direction
def h_spectral_zero(H):
    w, W = np.linalg.eig(H)
    w = np.maximum(w, 0)
    return W.dot(np.diag(w).dot(la.inv(W)))


H_f = h_spectral_flip(H)
H_z = h_spectral_zero(H)
H_c = h_spectral_cur(H)
H_cc = h_spectral_cc(H)

w, W = np.linalg.eig(H_cc)

H = np.array([[1, 0, 0],
              [0, 0.001, 0],
              [0, 0, 0.00001]], dtype=np.double)

w, W = np.linalg.eigh(H)
H_cc = h_spectral_cc(H, delta=1)
H_c = h_spectral_cur(H)