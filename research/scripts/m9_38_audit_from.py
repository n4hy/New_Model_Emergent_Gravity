#!/usr/bin/env python3
"""M9.38 audit. N=10, R=3, source (4,5,5), σ=0.9, α=0.03, seed 11.

Own M_hat, own n=65 DST Poisson. Tries to REFUTE C_newt.

Writes ../data/m9_38_audit_from.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, ALPHA, SEED = 10, 3, 0.9, 0.03, 11
SRC = (4, 5, 5)
LBOX, NBOX, GCONST = 1.0, 65, 1.0
PROBES = (0.30 * LBOX, 0.35 * LBOX, 0.40 * LBOX)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def poisson_blob(mass):
    xs = np.linspace(-LBOX, LBOX, NBOX)
    h = float(xs[1] - xs[0])
    rhs = np.zeros((NBOX, NBOX, NBOX))
    mid = NBOX // 2
    rhs[mid, mid, mid] = 4.0 * np.pi * GCONST * mass / (h**3)
    m = NBOX - 2
    fhat = rhs[1:-1, 1:-1, 1:-1]
    for ax in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=ax)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for ax in (0, 1, 2):
        phat = dst(phat, type=1, axis=ax)
    phi = np.zeros((NBOX, NBOX, NBOX))
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    return xs, phi


def interp(xs, field, p):
    def one(ax, val):
        k = int(np.searchsorted(ax, val) - 1)
        k = max(0, min(k, len(ax) - 2))
        t = (val - ax[k]) / (ax[k + 1] - ax[k])
        return k, t

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
                        j = idx(xx, yy, zz)
                        ham[i, j] = ham[j, i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros(vol)
    stag = np.zeros(vol)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (SIG * SIG))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    c1 = 0.5 * ((c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))) +
                (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))).T)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    r2max = RADIUS * RADIUS

    def S(c, sl):
        z = np.clip(np.linalg.eigvalsh(c[np.ix_(sl, sl)]), CLIP, 1 - CLIP)
        return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds, pflat, inside = [], [], []
    for cx, cy, cz in centers:
        sl, s_f = [], 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        sl.append(idx(x, y, z))
                        s_f += de[idx(x, y, z)]
        sl = np.array(sl, dtype=int)
        ds.append(S(c1, sl) - S(c0, sl))
        pflat.append(s_f)
        inside.append((cx - SRC[0]) ** 2 + (cy - SRC[1]) ** 2 + (cz - SRC[2]) ** 2 <= r2max)
    ds, pflat = np.asarray(ds, float), np.asarray(pflat, float)
    inside = np.asarray(inside, bool)
    well = inside & (np.abs(pflat) > 1e-6)
    rng = np.random.default_rng(SEED)
    widx = np.flatnonzero(well)
    rng.shuffle(widx)
    half = len(widx) // 2
    even, odd = widx[:half], widx[half:]
    kappa = float(np.median(ds[even] / pflat[even]))
    m_hat = float(np.median(ds[odd] / kappa))
    m_enc = float(np.median(pflat[odd]))
    xs, phi = poisson_blob(m_hat)
    h = float(xs[1] - xs[0])
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    gm = GCONST * m_hat
    accs, errs = [], []
    for r in PROBES:
        ar = interp(xs, ax, (r, 0.0, 0.0))
        accs.append(ar)
        errs.append(abs(abs(ar) * r * r / gm - 1.0))
    lr = np.log(np.array(PROBES, float))
    la = np.log(np.abs(np.array(accs, float)))
    coef, _, _, _ = np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)
    slope = float(coef[0])
    c_newt = bool(all(a < 0 for a in accs) and all(e < 0.05 for e in errs) and abs(slope + 2) < 0.08)
    c_mass = bool(abs(m_hat / m_enc - 1.0) < 0.10)
    payload = {
        "task": "m9.38_audit_from",
        "kappa": kappa,
        "M_hat": m_hat,
        "M_enc": m_enc,
        "mass_rel": abs(m_hat / m_enc - 1.0),
        "a_r": accs,
        "c1_ii": errs,
        "slope": slope,
        "C_mass": c_mass,
        "C_newt": c_newt,
        "verdicts": {
            "C_mass": "CONFIRMED" if c_mass else "REFUTED",
            "C_newt": "CONFIRMED" if c_newt else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_38_audit_from.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_mass and c_newt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
