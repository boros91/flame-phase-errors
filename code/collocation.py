"""Spectral collocation for the rescaled flame problem
    delta * y'(x) = y^2 - y^3,   x in [-1, 1],   y(-1) = delta,
(x = delta*t - 1).  Exact: y(x) = Y((x - x_half)/delta), x_half = delta*c(delta) - 1.
Nearest complex singularities of y:  x = delta*ln(a) +- i*pi*delta,  a = 1/delta - 1.
"""
import numpy as np
from flame import exact, shift, master


def cheb(N):
    """Chebyshev differentiation matrix and Gauss-Lobatto points (Trefethen),
    ordered from +1 to -1."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def sinh_map(x0, eps):
    """x = g(xi) = x0 + eps*sinh(A*xi + B) mapping [-1,1] -> [-1,1]
    (Tee & Trefethen 2006). Returns g, g'."""
    al = np.arcsinh((-1 - x0) / eps)
    be = np.arcsinh((1 - x0) / eps)
    A, B = (be - al) / 2, (be + al) / 2
    g = lambda xi: x0 + eps * np.sinh(A * xi + B)
    gp = lambda xi: eps * A * np.cosh(A * xi + B)
    return g, gp


def identity_map():
    return (lambda xi: xi), (lambda xi: np.ones_like(xi))


def x_half(delta):
    return delta * shift(delta) - 1.0


def singularity(delta):
    """Nearest complex singularities of y(x): x = delta*ln(a) +- i*pi*delta."""
    a = 1 / delta - 1
    return delta * np.log(a), np.pi * delta


def initial_guess(x, delta, xh):
    """Generic asymptotic shape: y ~ 1/(1+softplus(1-t)), t=(x-xh)/delta.
    (softplus is a crude two-sided approximation of the Wright omega function)."""
    t = (x - xh) / delta
    z = 1.0 - t
    sp = np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))
    return 1.0 / (1.0 + sp)


def setup(delta, N, mapping, x0=None, eps=None):
    """Nodes, differentiation matrix in x, and tau(xi) = (x - x_half)/delta computed
    without cancellation (needed to evaluate the exact solution accurately)."""
    D, xi = cheb(N)
    xh = x_half(delta)
    if mapping == 'sinh':
        al = np.arcsinh((-1 - x0) / eps); be = np.arcsinh((1 - x0) / eps)
        A, B = (be - al) / 2, (be + al) / 2
        g = lambda s: x0 + eps * np.sinh(A * s + B)
        gp = lambda s: eps * A * np.cosh(A * s + B)
        tau = lambda s: (x0 - xh) / delta + (eps / delta) * np.sinh(A * s + B)
    else:
        g = lambda s: s
        gp = lambda s: np.ones_like(s)
        tau = lambda s: (s - xh) / delta
    Dx = D / gp(xi)[:, None]
    return xi, g(xi), Dx, tau


def bary_row(xi_nodes, s):
    """Row vector l with l @ vals = interpolant(s)."""
    N = len(xi_nodes) - 1
    w = (-1.0) ** np.arange(N + 1); w[0] *= 0.5; w[-1] *= 0.5
    d = s - xi_nodes
    if np.any(d == 0):
        r = (d == 0).astype(float); return r
    c = w / d
    return c / c.sum()


def solve(delta, N, mapping='sinh', x0=None, eps=None, tol=1e-12, maxit=80, logvar=True,
          pin=False):
    """Damped Newton for the collocation equations. logvar=True solves for v = ln y:
       delta v' = e^v - e^{2v},  v(-1) = ln delta."""
    xi, x, Dx, tau = setup(delta, N, mapping, x0, eps)
    y = initial_guess(tau(xi))
    y[-1] = delta
    if logvar:
        u = np.log(y); bc = np.log(delta)
        F = lambda u: np.exp(u) - np.exp(2 * u)
        dF = lambda u: np.exp(u) - 2 * np.exp(2 * u)
    else:
        u = y.copy(); bc = delta
        F = lambda u: u**2 - u**3
        dF = lambda u: 2 * u - 3 * u**2

    if pin:
        # phase-pinned formulation: replace y(-1) = delta by y(x_half) = 1/2,
        # x_half known in closed form (translation property)
        if mapping == 'sinh':
            al = np.arcsinh((-1 - x0) / eps); be = np.arcsinh((1 - x0) / eps)
            A, B = (be - al) / 2, (be + al) / 2
            s_h = (np.arcsinh((x_half(delta) - x0) / eps) - B) / A
        else:
            s_h = x_half(delta)
        prow = bary_row(xi, s_h)
        pval = np.log(0.5) if logvar else 0.5

    def res(u):
        R = delta * Dx @ u - F(u)
        R[-1] = (prow @ u - pval) if pin else (u[-1] - bc)
        return R
    R = res(u); hist = []
    for it in range(maxit):
        J = delta * Dx - np.diag(dF(u))
        if pin:
            J[-1, :] = prow
        else:
            J[-1, :] = 0; J[-1, -1] = 1
        du = np.linalg.solve(J, -R)
        lam, r0 = 1.0, np.linalg.norm(R)
        while True:
            un = u + lam * du; Rn = res(un)
            if np.linalg.norm(Rn) < (1 - 1e-4 * lam) * r0 or lam < 1e-3:
                break
            lam /= 2
        u, R = un, Rn
        step = np.max(np.abs(lam * du)); hist.append(step)
        if step < tol:
            break
    y = np.exp(u) if logvar else u
    return dict(xi=xi, x=x, y=y, its=it + 1, converged=step < tol, tau=tau, last_step=step)


def initial_guess(tau):
    """Generic two-sided approximation y ~ 1/(1+softplus(1-tau)) (softplus(z) = ln(1+e^z)
    approximates the Wright omega function for z -> +-infinity)."""
    z = 1.0 - tau
    sp = np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))
    return 1.0 / (1.0 + sp)


def bary_eval(xi_nodes, vals, xi_eval):
    """Barycentric interpolation on Chebyshev-Lobatto nodes."""
    N = len(xi_nodes) - 1
    w = (-1.0) ** np.arange(N + 1); w[0] *= 0.5; w[-1] *= 0.5
    diff = xi_eval[:, None] - xi_nodes[None, :]
    hit = diff == 0
    diff[hit] = 1.0
    C = w / diff
    out = (C @ vals) / C.sum(axis=1)
    r, c = np.nonzero(hit)
    out[r] = vals[c]
    return out


def error(sol, M=4000):
    """Max error of the interpolant on M Chebyshev points of the first kind (in xi)."""
    xi_e = np.cos(np.pi * (np.arange(M) + 0.5) / M)
    p = bary_eval(sol['xi'], sol['y'], xi_e)
    return np.max(np.abs(p - master(sol['tau'](xi_e))))
