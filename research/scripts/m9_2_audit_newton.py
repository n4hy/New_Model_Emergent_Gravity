#!/usr/bin/env python3
"""M9.2 audit: different source (Gaussian) and different n.

No import of the solver. Own 7-point Poisson, own gradient
(grid central difference, then trilinear a). Tries to REFUTE C1–C5.

Writes ../data/m9_2_audit_newton.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
L, G, MASS, SIG = 1.0, 1.0, 1.0, 0.025
NS = (43, 61)
PROBES = (0.30 * L, 0.35 * L, 0.40 * L)


def xs_of(n):
    return np.linspace(-L, L, n)


def gauss_rho(xs):
    n = len(xs)
    h = float(xs[1] - xs[0])
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    dens = np.exp(-0.5 * (xx * xx + yy * yy + zz * zz) / (SIG * SIG))
    m = float(dens.sum() * h**3)
    dens *= MASS / m
    w = dens * h**3
    rms = float(np.sqrt(np.sum((xx * xx + yy * yy + zz * zz) * w) / MASS))
    return dens, rms


def _dst(rhs_int, h):
    axes = tuple(range(rhs_int.ndim))
    fhat = rhs_int
    for ax in axes:
        fhat = dst(fhat, type=1, axis=ax)
    m = rhs_int.shape[0]
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    if rhs_int.ndim == 3:
        lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    else:
        lam = lam1[:, None] + lam1[None, :]
    phat = fhat / lam
    for ax in axes:
        phat = dst(phat, type=1, axis=ax)
    return phat / (2.0 * (m + 1)) ** rhs_int.ndim


def solve3(xs, rhs):
    n = len(xs)
    h = float(xs[1] - xs[0])
    phi = np.zeros((n, n, n))
    phi[1:-1, 1:-1, 1:-1] = _dst(rhs[1:-1, 1:-1, 1:-1], h)
    return phi


def tril(xs, field, p):
    def one(ax, val):
        if val <= ax[0]:
            return 0, 0.0
        if val >= ax[-1]:
            return len(ax) - 2, 1.0
        k = int(np.searchsorted(ax, val) - 1)
        k = max(0, min(k, len(ax) - 2))
        return k, (val - ax[k]) / (ax[k + 1] - ax[k])

    i, tx = one(xs, p[0])
    j, ty = one(xs, p[1])
    k, tz = one(xs, p[2])
    acc = 0.0
    for di, wi in ((0, 1 - tx), (1, tx)):
        for dj, wj in ((0, 1 - ty), (1, ty)):
            for dk, wk in ((0, 1 - tz), (1, tz)):
                acc += wi * wj * wk * field[i + di, j + dj, k + dk]
    return float(acc)


def accel_phi(xs, phi, r):
    h = float(xs[1] - xs[0])
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    return tril(xs, phi, (r, 0.0, 0.0)), tril(xs, ax, (r, 0.0, 0.0))


def slope(radii, accs):
    accs = np.array(accs, float)
    if float(np.max(np.abs(accs))) == 0.0:
        return None
    lr = np.log(np.array(radii, float))
    la = np.log(np.abs(accs))
    c, _, _, _ = np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)
    return float(c[0])


def run(n, gconst, rho):
    xs = xs_of(n)
    phi = solve3(xs, 4.0 * np.pi * gconst * rho)
    gm = abs(gconst) * MASS
    phis, accs, e1, e2 = [], [], [], []
    for r in PROBES:
        p, a = accel_phi(xs, phi, r)
        phis.append(p)
        accs.append(a)
        e1.append(abs(abs(a) * r * r / gm - 1.0))
        e2.append(abs(p * r / (gconst * MASS) + 1.0))
    sl = slope(PROBES, accs)
    return {
        "n": n,
        "a_r": accs,
        "phi": phis,
        "c1_ii": e1,
        "c2": e2,
        "slope": sl,
        "c1_i": all((a < 0) if gconst > 0 else (a > 0) for a in accs),
        "c1_ii_pass": all(v < 0.05 for v in e1),
        "c1_iii_pass": bool(sl is not None and abs(sl + 2.0) < 0.08),
        "c2_pass": all(v < 0.05 for v in e2),
        "max_a": float(np.max(np.abs(accs))),
        "max_p": float(np.max(np.abs(phis))),
    }


def run2d(n):
    xs = xs_of(n)
    rho, _ = gauss_rho(xs)
    mid = n // 2
    sig = rho[:, :, mid]
    h = float(xs[1] - xs[0])
    phi = np.zeros((n, n))
    phi[1:-1, 1:-1] = _dst(4 * np.pi * G * sig[1:-1, 1:-1], h)
    accs, e1 = [], []
    m2 = float(sig.sum() * h * h)
    for r in PROBES:
        # 2d: differentiate along x in the plane
        ax = np.zeros_like(phi)
        ax[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * h)
        # interpolate ax at (r,0)
        def one(axx, val):
            k = int(np.searchsorted(axx, val) - 1)
            k = max(0, min(k, len(axx) - 2))
            t = (val - axx[k]) / (axx[k + 1] - axx[k])
            return k, t
        i, tx = one(xs, r)
        j, ty = one(xs, 0.0)
        a = float(
            (1 - tx) * (1 - ty) * ax[i, j]
            + tx * (1 - ty) * ax[i + 1, j]
            + (1 - tx) * ty * ax[i, j + 1]
            + tx * ty * ax[i + 1, j + 1]
        )
        accs.append(a)
        e1.append(abs(abs(a) * r * r / (G * m2) - 1.0))
    sl = slope(PROBES, accs)
    return {
        "slope": sl,
        "c1_ii_pass": all(v < 0.05 for v in e1),
        "c4_alpha": abs(sl + 1.0) < 0.15,
        "c1_ii": e1,
    }


def main() -> int:
    fine_n = NS[1]
    xs = xs_of(fine_n)
    rho, rms = gauss_rho(xs)
    xs_c = xs_of(NS[0])
    rho_c, rms_c = gauss_rho(xs_c)
    fine = run(fine_n, G, rho)
    coarse = run(NS[0], G, rho_c)
    vac = run(fine_n, G, np.zeros_like(rho))
    flip = run(fine_n, -G, rho)
    mut = run2d(fine_n)
    c1 = bool(fine["c1_i"] and fine["c1_ii_pass"] and fine["c1_iii_pass"])
    c2 = bool(fine["c2_pass"])
    c3 = bool(vac["max_a"] < 1e-8 * max(fine["max_a"], 1e-30) and vac["max_p"] < 1e-8 * max(fine["max_p"], 1e-30))
    c4 = bool(mut["c4_alpha"] and not mut["c1_ii_pass"])
    c5 = bool(flip["c1_i"] and flip["c1_ii_pass"] and (not all(a < 0 for a in flip["a_r"])))
    payload = {
        "task": "m9.2_audit_newton",
        "source": "gaussian",
        "rms": [rms_c, rms],
        "coarse": coarse,
        "fine": fine,
        "vacuum": vac,
        "flipG": flip,
        "mutation_2d": mut,
        "verdicts": {
            "C1": "CONFIRMED" if c1 else "REFUTED",
            "C2": "CONFIRMED" if c2 else "REFUTED",
            "C3": "CONFIRMED" if c3 else "REFUTED",
            "C4": "CONFIRMED" if c4 else "REFUTED",
            "C5": "CONFIRMED" if c5 else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_2_audit_newton.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
