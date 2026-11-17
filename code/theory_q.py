"""Symbolic local-error coefficients for f(y) = y^q (1-y), general integer q >= 2.
Checks the scaling C(y) = y^{1+(p+1)(q-1)} E(0,y) used in Theorem 3.2 and returns the
delay density g = -C/f^2 together with an antiderivative G."""
import sympy as sp

y, h = sp.symbols('y h', positive=True)


def setup(q):
    f = y**q * (1 - y)
    F = sp.Lambda(y, y**q * (1 - y))
    return f, F


def exact_flow(f, order):
    terms, D = y, f
    for k in range(1, order + 1):
        terms += h**k / sp.factorial(k) * D
        D = sp.diff(D, y) * f
    return sp.expand(terms)


def lead_error(Phi, f, maxorder=5):
    diff = sp.expand(sp.series(Phi - exact_flow(f, maxorder), h, 0, maxorder + 1).removeO())
    for k in range(1, maxorder + 1):
        c = sp.simplify(diff.coeff(h, k))
        if c != 0:
            return k - 1, sp.factor(c)


def methods(q):
    f, F = setup(q)
    M = {'Euler': y + h * f}
    Yi = y
    for _ in range(7):
        Yi = sp.series(y + h * F(Yi), h, 0, 7).removeO()
    M['Implicit Euler'] = sp.expand(Yi)
    k1 = F(y); k2 = F(y + h * k1)
    M['Heun'] = y + h / 2 * (k1 + k2)
    k1 = F(y); k2 = F(y + h / 2 * k1); k3 = F(y + h / 2 * k2); k4 = F(y + h * k3)
    M['RK4'] = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return f, M


if __name__ == '__main__':
    for q in [2, 3]:
        f, M = methods(q)
        print(f'================ q = {q}')
        for name, Phi in M.items():
            p, C = lead_error(Phi, f)
            expo = 1 + (p + 1) * (q - 1)
            E0 = sp.simplify(C / y**expo)
            g = sp.factor(sp.simplify(-C / f**2))
            G = sp.simplify(sp.integrate(sp.apart(g, y), y))
            kappa = sp.limit(g * y, y, 0) if p == 1 else None
            print(f'{name}: p={p}; C = y^{expo} * ({E0});  E0 polynomial: {E0.is_polynomial(y)}')
            print(f'   g = {g}\n   G = {G}\n   kappa = {kappa}')
