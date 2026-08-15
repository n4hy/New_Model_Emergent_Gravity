#!/usr/bin/env python3
"""M9.42: unequal masses. Interior null is M/r², not the CM.

Paper 50 cancelled at the midpoint because M_A ≈ M_B.
That is a symmetry. Between two attractors the force
balances where M_A/r_A² = M_B/r_B², closer to the
lighter mass — not at the centre of mass.

PRE-REGISTERED:
  N=12, A=(2,6,6), B=(10,6,6), σ=1. Wider gap.
  α_A=0.02, α_B=0.05. Orthonormal two-source C.
  M_A, M_B = ∑δe of the single-packet states.
  Midpoint at 0, SEP=0.50 L, x_A=−SEP/2, x_B=+SEP/2.
  x_cm    = (M_A x_A + M_B x_B) / (M_A + M_B)
  x_force : M_A/(x−x_A)² = M_B/(x_B−x)², x in (x_A, x_B)
  C_add   |M_AB − M_A − M_B| / |M_AB| < 0.08
  C_uneq  |M_B / M_A − 1| > 0.40
  C_force PRIMARY. interpolated a_x=0 within 0.04 L of x_force
  C_notcm |x_null − x_cm| > |x_null − x_force|
          (null is the inverse-square point, not the CM)
  C_side  x_null has the same sign as x_force (toward the light mass)

Inherited Poisson. Not a derivation.

Writes ../data/m9_42_bary.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N = 12
A = (2, 6, 6)
B = (10, 6, 6)
MID = (6, 6, 6)
SIGMA = 1.0
ALPHA_A = 0.02
ALPHA_B = 0.05
LBOX = 1.0
NBOX = 65
GCONST = 1.0
SEP = 0.50 * LBOX


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham[i, j] = ham[j, i] = -1.0
    return ham


def raw_packet(uo, uu, n, src, sigma):
    env = np.zeros((n**3,), dtype=float)
    stag = np.zeros((n**3,), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (sigma * sigma))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    return uo @ (uo.T @ env), uu @ (uu.T @ stag)


def orthonormalize(v1, v2):
    e1 = v1 / np.linalg.norm(v1)
    v2 = v2 - e1 * np.dot(e1, v2)
    return e1, v2 / np.linalg.norm(v2)


def one_source(uo, uu, c0, n, src, sigma, alpha):
    left, right = raw_packet(uo, uu, n, src, sigma)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return 0.5 * (corr + corr.T)


def two_source(uo, uu, c0, n, src_a, src_b, sigma, aa, ab):
    la, ra = raw_packet(uo, uu, n, src_a, sigma)
    lb, rb = raw_packet(uo, uu, n, src_b, sigma)
    la, lb = orthonormalize(la, lb)
    ra, rb = orthonormalize(ra, rb)
    corr = (
        c0
        + aa * (np.outer(ra, ra) - np.outer(la, la))
        + ab * (np.outer(rb, rb) - np.outer(lb, lb))
    )
    return 0.5 * (corr + corr.T)


def dst_poisson(rhs):
    nbox = rhs.shape[0]
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    m = nbox - 2
    fhat = rhs[1:-1, 1:-1, 1:-1].copy()
    for ax in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=ax)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for ax in (0, 1, 2):
        phat = dst(phat, type=1, axis=ax)
    phi = np.zeros((nbox, nbox, nbox), dtype=float)
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    return xs, h, phi


def interp(xs, field, p):
    def one(ax, val):
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


def deposit(de, n, mid, scale, nbox):
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    rhs = np.zeros((nbox, nbox, nbox), dtype=float)
    mx, my, mz = mid
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                px = (x - mx) * scale
                py = (y - my) * scale
                pz = (z - mz) * scale
                if max(abs(px), abs(py), abs(pz)) >= LBOX - 0.5 * h:
                    continue
                ix = int(np.argmin(np.abs(xs - px)))
                iy = int(np.argmin(np.abs(xs - py)))
                iz = int(np.argmin(np.abs(xs - pz)))
                if min(ix, iy, iz) <= 0 or max(ix, iy, iz) >= nbox - 1:
                    continue
                rhs[ix, iy, iz] += 4.0 * np.pi * GCONST * de[i] / (h**3)
    return rhs


def main() -> int:
    ham = hop_H(N)
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T
    ca = one_source(uo, uu, c0, N, A, SIGMA, ALPHA_A)
    cb = one_source(uo, uu, c0, N, B, SIGMA, ALPHA_B)
    cab = two_source(uo, uu, c0, N, A, B, SIGMA, ALPHA_A, ALPHA_B)
    de_a = np.sum(ham * ca, axis=1) - np.sum(ham * c0, axis=1)
    de_b = np.sum(ham * cb, axis=1) - np.sum(ham * c0, axis=1)
    de_ab = np.sum(ham * cab, axis=1) - np.sum(ham * c0, axis=1)
    m_a, m_b, m_ab = float(de_a.sum()), float(de_b.sum()), float(de_ab.sum())
    add_rel = abs(m_ab - m_a - m_b) / abs(m_ab)
    c_add = bool(add_rel < 0.08)
    c_uneq = bool(abs(m_b / m_a - 1.0) > 0.40)

    scale = SEP / float(B[0] - A[0])
    x_a, x_b = -0.5 * SEP, 0.5 * SEP
    x_cm = (m_a * x_a + m_b * x_b) / (m_a + m_b)
    # M_A / (x-x_A)^2 = M_B / (x_B-x)^2, take the root in (x_A, x_B)
    # rA / rB = sqrt(M_A / M_B)
    rat = np.sqrt(m_a / m_b)
    x_force = (x_a + rat * x_b) / (1.0 + rat)
    rhs = deposit(de_ab, N, MID, scale, NBOX)
    xs, h, phi = dst_poisson(rhs)
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)

    grid = np.linspace(x_a + 0.05 * SEP, x_b - 0.05 * SEP, 121)
    vals = np.array([interp(xs, ax, (x, 0.0, 0.0)) for x in grid])
    x_null = None
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0.0:
            t = vals[i] / (vals[i] - vals[i + 1])
            x_null = float(grid[i] + t * (grid[i + 1] - grid[i]))
            break
    force_err = abs(x_null - x_force) if x_null is not None else None
    cm_err = abs(x_null - x_cm) if x_null is not None else None
    c_force = bool(force_err is not None and force_err < 0.04)
    c_notcm = bool(
        x_null is not None and force_err is not None and cm_err is not None and cm_err > force_err
    )
    c_side = bool(x_null is not None and np.sign(x_null) == np.sign(x_force))

    ok = bool(c_add and c_uneq and c_force and c_notcm and c_side)
    payload = {
        "task": "m9.42_bary",
        "M_A": m_a,
        "M_B": m_b,
        "M_AB": m_ab,
        "mass_ratio": m_b / m_a,
        "add_rel": add_rel,
        "x_A": x_a,
        "x_B": x_b,
        "x_cm": x_cm,
        "x_force": x_force,
        "x_null": x_null,
        "force_err": force_err,
        "cm_err": cm_err,
        "C_add": c_add,
        "C_uneq": c_uneq,
        "C_force_PRIMARY": c_force,
        "C_notcm": c_notcm,
        "C_side": c_side,
        "all_gates": ok,
        "verdict": "INVSQ_NULL" if ok else "BARY_FAIL",
        "not_claimed": ["derived Poisson", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_42_bary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
