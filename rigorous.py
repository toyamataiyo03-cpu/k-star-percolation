"""Computer-assisted rigorous constants for Theorem 1 (a = 1).

Everything below is either exact symbolic algebra (sympy) or ball arithmetic (Arb via python-flint)
over boxes covering the whole parameter domain, so every printed bound is a rigorous upper bound.

Domain: 0 < eps <= EPS0, r = 2(1-eps), t = sqrt(eps), u = t v, 0 <= u <= U(eps) = arccos(1-eps).
"""
import sympy as sp
from flint import arb, ctx
ctx.prec = 100

EPS0 = sp.Rational(1, 10)
T0 = sp.sqrt(EPS0)
X0 = sp.Rational(55, 100)          # arcsin polynomial bounds are used on [0, X0]
VMAX = sp.Rational(143, 100)       # claim: U(eps) <= VMAX*sqrt(eps)

v, t, u, x = sp.symbols("v t u x", nonnegative=True)
A = 1 / (1 - t**2)

# ---- elementary polynomial bounds -------------------------------------------------
s_up = u - sp.Rational(2, 3) * u**3 + sp.Rational(2, 15) * u**5          # >= sin(2u)/2 for u>=0
s_lo = s_up - sp.Rational(4, 315) * u**7                                  # <= sin(2u)/2 for u>=0
a_lo = x + x**3 / 6 + sp.Rational(3, 40) * x**5                           # <= asin x on [0,1)
x0a = arb(str(X0.p)) / arb(str(X0.q))
R7 = (x0a.asin() - x0a - x0a**3 / 6 - arb(3) / 40 * x0a**5) / x0a**7      # asin x <= a_lo + R7 x^7 on [0,X0]
R7u = arb(R7.upper())
R7s = sp.Symbol("R7")
a_up = a_lo + R7s * x**7

g_lo = 2 * (a_lo.subs(x, A * s_lo) - u)
g_up = 2 * (a_up.subs(x, A * s_up) - u)

G0 = v * (2 - v**2)
G1 = v * (2 - v**2 / 3 - v**4 / 4)
v0 = sp.sqrt(sp.Rational(2, 3))
c = sp.nsimplify(G0.subs(v, v0))
assert sp.simplify(G1.subs(v, v0) - sp.Rational(5, 4) * c) == 0


def scaled_remainder(g):
    P = sp.together(sp.expand(g.subs(u, t * v)) / t**3)
    Bexpr = sp.together((P - G0 - t**2 * G1) / t**4)
    num, den = sp.fraction(sp.factor(Bexpr))
    # sanity: no negative powers of t remain
    assert sp.Poly(den, t, v).as_expr().subs(t, 0) != 0
    return sp.expand(num), sp.expand(den)


Blo_n, Blo_d = scaled_remainder(g_lo)
Bup_n, Bup_d = scaled_remainder(g_up)
w0 = sp.Symbol('w0')
# (G1(v)-G1(w0))/(v-w0) written out exactly as a polynomial in v and w0 = v0
H = 2 - (v**2 + v*w0 + w0**2)/3 - (v**4 + v**3*w0 + v**2*w0**2 + v*w0**3 + w0**4)/4
assert sp.expand(H*(v - w0) - (G1 - G1.subs(v, w0))) == 0


def eval_arb(expr, env):
    """Rigorous evaluation of a sympy expression in ball arithmetic (rationals exact, sqrt via Arb)."""
    if expr.is_Symbol:
        return env[expr]
    if expr.is_Integer:
        return arb(int(expr))
    if expr.is_Rational:
        return arb(int(expr.p)) / arb(int(expr.q))
    if expr.is_Add:
        out = arb(0)
        for a in expr.args:
            out += eval_arb(a, env)
        return out
    if expr.is_Mul:
        out = arb(1)
        for a in expr.args:
            out *= eval_arb(a, env)
        return out
    if expr.is_Pow:
        b, e = expr.args
        bb = eval_arb(b, env)
        if e.is_Integer:
            return bb ** int(e)
        if e == sp.Rational(1, 2):
            return bb.sqrt()
        if e == -sp.Rational(1, 2):
            return 1 / bb.sqrt()
    raise ValueError(f"unsupported: {expr}")


def ball(lo, hi):
    m = (lo + hi) / 2
    r = (hi - lo) / 2
    return m + arb(0, float(r.upper()) * (1 + 1e-9))


def sup_abs_box(num, den, nv=60, nt=60, vmax=VMAX, fix_v=None):
    """Rigorous upper bound of |num/den| over v in [0,vmax], t in [0,T0] (box subdivision)."""
    t0 = arb(1) / arb(10).sqrt() * (1 + arb("1e-12"))     # >= sqrt(0.1)
    vm = arb(int(vmax.p)) / arb(int(vmax.q))
    num = sp.expand(num); den = sp.expand(den)
    best = 0.0
    vs = [fix_v] if fix_v is not None else [ball(vm * i / nv, vm * (i + 1) / nv) for i in range(nv)]
    for vb in vs:
        for j in range(nt):
            tb = ball(t0 * j / nt, t0 * (j + 1) / nt)
            env = {v: vb, t: tb, R7s: R7u, sp.Symbol('w0'): (arb(2)/3).sqrt()}
            dd = eval_arb(den, env)
            assert not dd.contains(0), "denominator may vanish"
            val = abs(eval_arb(num, env)) / abs(dd)
            best = max(best, float(val.upper()))
    return arb(best) * (1 + arb("1e-9"))


v0a = (arb(2) / 3).sqrt()
ca = 4 * arb(6).sqrt() / 9
B_lo_v0 = sup_abs_box(Blo_n, Blo_d, fix_v=v0a)
B_up = sup_abs_box(Bup_n, Bup_d)
L = sup_abs_box(sp.expand(H), sp.Integer(1), nt=1)
Kdelta = max(B_lo_v0, B_up + L * L / (8 * v0a), key=float)
print("R7 <=", R7u)
print("sup|B_lo(v0,t)| <=", B_lo_v0, "  sup|B_up| <=", B_up, "  L <=", L)
print("=> |delta_max/eps^1.5 - c(1+5eps/4)| <= K_delta eps^2,  K_delta <=", arb(Kdelta.upper()))

# ---- validity checks of the elementary bounds on the whole domain -------------------------
eps0 = arb(1) / 10
Amax = 1 / (1 - eps0)
Umax = arb(143) / 100 * eps0.sqrt()
xmax = Amax * Umax                      # w = A sin(2u)/2 <= A u <= A U
print("max argument of asin <=", arb(xmax.upper()), "<= X0 =", float(X0), bool(xmax < x0a))
# U(eps) <= VMAX sqrt(eps):  cos(V t) <= 1 - V^2 t^2/2 + V^4 t^4/24 <= 1 - t^2  iff  V^2/2 - 1 >= V^4 t^2/24
V = arb(143) / 100
print("U <= 1.43 sqrt(eps) on eps<=0.1:", bool(V**2 / 2 - 1 >= V**4 * eps0 / 24))
# concavity: g'' <= 0  iff  A^2 cos^2(2u) <= 4(1-w^2); suffices A^2 <= 4(1 - X0^2)
print("g concave:", bool(Amax**2 < 4 * (1 - x0a**2)))
# |g''| <= u * Mu with Mu = 2A(4/sqrt(1-X0^2) + A^2/(1-X0^2)^{3/2});  u <= VMAX sqrt(eps)
Mu = 2 * Amax * (4 / (1 - x0a**2).sqrt() + Amax**2 / (1 - x0a**2) ** arb(1.5))
M = arb((Mu * V).upper())
print("|g''| <= M sqrt(eps),  M <=", M)

# ---- the one-dimensional integral F(e) (Theorem 1 main term) ----------------------------
# F(e) = int_0^{s1} (1-e s)(1 - s^{3/2} - (5/4) e s^{5/2}) ds,  F(0)=3/5, F'(0)=-4/7,  |F''| <= 5/9 + 25/24
KF = (arb(5) / 9 + arb(25) / 24) / 2
print("K_F <=", arb(KF.upper()))

# ---- Theorem 1/2 explicit constants ------------------------------------------------------
pi = arb.pi()
lam = arb(11) / 10
Kd = arb(Kdelta.upper())
KD = 8 * pi * 4 * pi**2 * (KF / (4 * ca**2) + Kd * lam**arb(4.5) / (18 * ca**3) + M * lam**arb(1.5) / (12 * ca))
print("K_D <=", arb(KD.upper()))
def ok(k):
    h = 2 * pi / k
    es = (pi / (ca * k)) ** (arb(2) / 3)
    cond_dom = lam * es <= eps0
    cover = lam**arb(1.5) - Kd * lam**arb(3.5) * es**2 / ca - h * M * (lam * es).sqrt() / 4
    return bool(cond_dom) and bool(cover >= 1), es
k0 = next(k for k in range(3, 100000) if ok(k)[0])
print("k0 =", k0, " eps* at k0 =", ok(k0)[1])
import json
json.dump({"EPS0": 0.1, "X0": 0.55, "VMAX": 1.43, "R7": float(R7u.upper()), "K_delta": float(Kdelta.upper()),
           "M": float(M.upper()), "K_F": float(KF.upper()), "B_lo_v0": float(B_lo_v0.upper()),
           "B_up": float(B_up.upper()), "L": float(L.upper()), "K_D": float(KD.upper()), "k0": k0, "lambda": 1.1},
          open("data/rigorous_constants.json", "w"), indent=1)
