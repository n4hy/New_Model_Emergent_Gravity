#!/usr/bin/env python3
"""M9.42 audit. N=10, A=(2,5,5), B=(8,5,5), α_A=0.02, α_B=0.04, σ=0.9.

Own masses. Interior null vs M/r², not CM. Tries to REFUTE C_force.

Writes ../data/m9_42_audit_bary.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N, SIG = 10, 0.9
A, B, MID = (2, 5, 5), (8, 5, 5), (5, 5, 5)
AA, AB = 0.02, 0.04
LBOX, NBOX, GCONST = 1.0, 65, 1.0
SEP = 0.50 * LBOX


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def raw_packet(uo, uu, n, src, sigma):
    env = np.zeros(n**3)
    stag = np.zeros(n**3)
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


def main() -> int:
    vol = N**3
    ham = np.zeros((vol, vol))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        ham[i, idx(xx, yy, zz)] = ham[idx(xx, yy, zz), i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T

    def one(src, alpha):
        L, R = raw_packet(uo, uu, N, src, SIG)
        L, R = L / np.linalg.norm(L), R / np.linalg.norm(R)
        c = c0 + alpha * (np.outer(R, R) - np.outer(L, L))
        return 0.5 * (c + c.T)

    ca, cb = one(A, AA), one(B, AB)
    la, ra = raw_packet(uo, uu, N, A, SIG)
    lb, rb = raw_packet(uo, uu, N, B, SIG)
    la, lb = orthonormalize(la, lb)
    ra, rb = orthonormalize(ra, rb)
    cab = c0 + AA * (np.outer(ra, ra) - np.outer(la, la)) + AB * (
        np.outer(rb, rb) - np.outer(lb, lb)
    )
    cab = 0.5 * (cab + cab.T)
    de_a = np.sum(ham * ca, axis=1) - np.sum(ham * c0, axis=1)
    de_b = np.sum(ham * cb, axis=1) - np.sum(ham * c0, axis=1)
    de_ab = np.sum(ham * cab, axis=1) - np.sum(ham * c0, axis=1)
    m_a, m_b, m_ab = float(de_a.sum()), float(de_b.sum()), float(de_ab.sum())
    scale = SEP / float(B[0] - A[0])
    x_a, x_b = -0.5 * SEP, 0.5 * SEP
    x_cm = (m_a * x_a + m_b * x_b) / (m_a + m_b)
    rat = np.sqrt(m_a / m_b)
    x_force = (x_a + rat * x_b) / (1.0 + rat)
    xs = np.linspace(-LBOX, LBOX, NBOX)
    h = float(xs[1] - xs[0])
    rhs = np.zeros((NBOX, NBOX, NBOX))
    mx, my, mz = MID
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                px, py, pz = (x - mx) * scale, (y - my) * scale, (z - mz) * scale
                if max(abs(px), abs(py), abs(pz)) >= LBOX - 0.5 * h:
                    continue
                ix = int(np.argmin(np.abs(xs - px)))
                iy = int(np.argmin(np.abs(xs - py)))
                iz = int(np.argmin(np.abs(xs - pz)))
                if min(ix, iy, iz) <= 0 or max(ix, iy, iz) >= NBOX - 1:
                    continue
                rhs[ix, iy, iz] += 4.0 * np.pi * GCONST * de_ab[i] / (h**3)
    m = NBOX - 2
    fhat = rhs[1:-1, 1:-1, 1:-1].copy()
    for axv in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=axv)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for axv in (0, 1, 2):
        phat = dst(phat, type=1, axis=axv)
    phi = np.zeros((NBOX, NBOX, NBOX))
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
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
    c_notcm = bool(x_null is not None and force_err is not None and cm_err > force_err)
    c_side = bool(x_null is not None and np.sign(x_null) == np.sign(x_force))
    payload = {
        "task": "m9.42_audit_bary",
        "M_A": m_a,
        "M_B": m_b,
        "mass_ratio": m_b / m_a,
        "x_cm": x_cm,
        "x_force": x_force,
        "x_null": x_null,
        "force_err": force_err,
        "cm_err": cm_err,
        "C_force": c_force,
        "C_notcm": c_notcm,
        "C_side": c_side,
        "verdicts": {
            "C_force": "CONFIRMED" if c_force else "REFUTED",
            "C_notcm": "CONFIRMED" if c_notcm else "REFUTED",
            "C_side": "CONFIRMED" if c_side else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_42_audit_bary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_force and c_side) else 1


if __name__ == "__main__":
    raise SystemExit(main())
