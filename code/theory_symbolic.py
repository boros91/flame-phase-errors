"""Symbolic derivation of the leading local-error coefficient C(y) and of the
delay density g(y) = -C(y)/f(y)^2 for one-step methods applied to the flame
equation y' = f(y) = y^2 - y^3.

Delay (formal, h -> 0):  D_h(delta, eta) ~ h^p * int_delta^eta g(y) dy.
"""
import sympy as sp

y, h, a, b, Y1 = sp.symbols('y h a b Y1', real=True)
f = y**2 - y**3
fp = sp.diff(f, y)


def exact_flow(order):
    """Taylor series of the exact flow phi_h(y) up to h^order."""
    terms, D = y, f
    for k in range(1, order + 1):
        terms += h**k / sp.factorial(k) * D
        D = sp.diff(D, y) * f
    return sp.expand(terms)


def lead_error(Phi, maxorder=5):
    """Return (p, C) with Phi - phi_h = h^{p+1} C(y) + O(h^{p+2})."""
    diff = sp.series(Phi - exact_flow(maxorder), h, 0, maxorder + 1).removeO()
    diff = sp.expand(diff)
    for k in range(1, maxorder + 1):
        c = sp.simplify(diff.coeff(h, k))
        if c != 0:
            return k - 1, sp.factor(c)
    raise RuntimeError


def rk(stages_fn):
    return stages_fn


F = sp.Lambda(y, y**2 - y**3)
methods = {}
methods['Euler'] = y + h * f
# implicit Euler: Y1 = y + h f(Y1); series solution by fixed-point iteration
Yi = y
for _ in range(6):
    Yi = sp.series(y + h * F(Yi), h, 0, 6).removeO()
methods['Implicit Euler'] = sp.expand(Yi)
# NSFD (3) of Abelman-Patidar, denominator function phi(h)=h and 1-exp(-h)
for name, ph in [('NSFD3, phi=h', h), ('NSFD3, phi=1-e^-h', 1 - sp.exp(-h))]:
    methods[name] = (1 + 2 * ph * y) * y / (1 + y * (1 + y) * ph)
for name, ph in [('NSFD4(a=1,b=-3), phi=h', h), ('NSFD4(a=1,b=-3), phi=1-e^-h', 1 - sp.exp(-h))]:
    aa, bb = 1, -3
    methods[name] = (y + ph * (aa * y**2 - bb * y**3)) / (1 - ph * (1 - aa) * y + ph * (1 - bb) * y**2)
# Heun (explicit trapezoid)
k1 = F(y); k2 = F(y + h * k1)
methods['Heun'] = y + h / 2 * (k1 + k2)
# classical RK4
k1 = F(y); k2 = F(y + h / 2 * k1); k3 = F(y + h / 2 * k2); k4 = F(y + h * k3)
methods['RK4'] = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == '__main__':
    out = {}
    for name, Phi in methods.items():
        p, C = lead_error(Phi)
        g = sp.simplify(-C / f**2)
        # behaviour near y = 0
        g0 = sp.limit(g * y**(2 - p), y, 0)  # g ~ g0 * y^(p-2)
        G = sp.simplify(sp.integrate(sp.apart(g, y), y))
        print(f'--- {name}: order p={p}')
        print('   C(y)   =', C)
        print('   g(y)   =', sp.factor(g))
        print(f'   g ~ {g0} * y^{p-2} as y->0')
        print('   int g  =', G)
        out[name] = (p, C, g, G)
