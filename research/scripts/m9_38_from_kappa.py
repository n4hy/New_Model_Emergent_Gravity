#!/usr/bin/env python3
"""M9.38: δS → M_hat via κ → inherited Newton.

Paper 47: κ weighs enclosed mass and points at the source.
This run feeds M_hat into the M9.2 DST Poisson solver.

PRE-REGISTERED:
  Fermion: N=12, R=3, packet (6,6,6), σ=1.0, α=0.02.
  κ from even well-inside split (seed 38).
  M_hat = median(δS/κ) on the odd well-inside subset.
  C_vac   |ρ(δS, Tr(K_vac ΔC))| > 0.95
  C_mass  |M_hat/M_enc − 1| < 0.10
  C_loc   Chebyshev(x_hat, x_true) ≤ 1
          (Paper 47 auditor was off by one; R=3 tomography)
  C_newt  PRIMARY. DST Poisson of a compact blob of mass
          M_hat at the centre of a Dirichlet cube, n=65,
          L=1, G=1. Probes r=0.30L, 0.35L, 0.40L:
          (i) a·rhat < 0
          (ii) ||a| r²/(G M_hat) − 1| < 0.05 at all three
          (iii) log-log slope α of |a| vs r has |α+2| < 0.08
  C_newt is inherited Einstein sourced by an entanglement
  mass. It is not a derivation of Poisson.

Not claimed: 8πG from κ, 1/4G, FGHMV, de Sitter, MODELS.md.

Writes ../data/m9_38_from_kappa.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
RADIUS = 3
SRC = (6, 6, 6)
SIGMA = 1.0
ALPHA = 0.02
SEED = 38
HIGH = 0.70
LBOX = 1.0
NBOX = 65
GCONST = 1.0
PROBES = (0.30 * LBOX, 0.35 * LBOX, 0.40 * LBOX)


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


def occupation_transfer(ham, n, src, sigma, alpha):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
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
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return c0, 0.5 * (corr + corr.T)


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr, sl):
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def in_ball(src, center, radius):
    return sum((src[k] - center[k]) ** 2 for k in range(3)) <= radius * radius


def poisson_blob(nbox, mass):
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    rhs = np.zeros((nbox, nbox, nbox), dtype=float)
    mid = nbox // 2
    rhs[mid, mid, mid] = 4.0 * np.pi * GCONST * mass / (h**3)
    m = nbox - 2
    fhat = rhs[1:-1, 1:-1, 1:-1]
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
    return xs, phi


def interp(xs, field, p):
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


def newton_c1(mass):
    xs, phi = poisson_blob(NBOX, mass)
    h = float(xs[1] - xs[0])
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    gm = GCONST * mass
    accs, errs = [], []
    for r in PROBES:
        ar = interp(xs, ax, (r, 0.0, 0.0))
        accs.append(ar)
        errs.append(abs(abs(ar) * r * r / gm - 1.0))
    lr = np.log(np.asarray(PROBES, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    coef, _, _, _ = np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)
    slope = float(coef[0])
    return {
        "a_r": accs,
        "c1_ii": errs,
        "slope": slope,
        "attractive": all(a < 0.0 for a in accs),
        "c1_ii_pass": all(e < 0.05 for e in errs),
        "c1_iii_pass": abs(slope + 2.0) < 0.08,
        "pass": all(a < 0.0 for a in accs)
        and all(e < 0.05 for e in errs)
        and abs(slope + 2.0) < 0.08,
    }


def main() -> int:
    ham = hop_H(N)
    c0, c1 = occupation_transfer(ham, N, SRC, SIGMA, ALPHA)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds, pflat, pk, inside = [], [], [], []
    for cx, cy, cz in centers:
        sl = np.array(
            [
                idx(x, y, z)
                for x in range(N)
                for y in range(N)
                for z in range(N)
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2max
            ],
            dtype=int,
        )
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pk.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
        pflat.append(float(np.sum(de[sl])))
        inside.append(in_ball(SRC, (cx, cy, cz), RADIUS))
    ds, pflat, pk = map(np.asarray, (ds, pflat, pk))
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
    mass_rel = abs(m_hat / m_enc - 1.0) if m_enc else None
    thresh = HIGH * float(np.max(ds))
    high = ds > thresh
    votes = np.zeros((N, N, N), dtype=int)
    survive = np.ones((N, N, N), dtype=bool)
    first = True
    for k, (cx, cy, cz) in enumerate(centers):
        if not high[k]:
            continue
        mask = np.zeros((N, N, N), dtype=bool)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2max:
                        mask[x, y, z] = True
                        votes[x, y, z] += 1
        survive = mask if first else (survive & mask)
        first = False
    if not first and np.any(survive):
        loc = np.unravel_index(int(np.argmax(np.where(survive, votes, -1))), votes.shape)
    else:
        loc = np.unravel_index(int(np.argmax(votes)), votes.shape)
    loc = (int(loc[0]), int(loc[1]), int(loc[2]))
    dist = max(abs(loc[k] - SRC[k]) for k in range(3))
    newt = newton_c1(m_hat)
    c_vac = bool(abs(pearson(ds, pk)) > 0.95)
    c_mass = bool(mass_rel is not None and mass_rel < 0.10)
    c_loc = bool(dist <= 1)
    c_newt = bool(newt["pass"])
    ok = bool(c_vac and c_mass and c_loc and c_newt)
    payload = {
        "task": "m9.38_from_kappa",
        "kappa": kappa,
        "M_enc": m_enc,
        "M_hat": m_hat,
        "mass_rel": mass_rel,
        "x_true": list(SRC),
        "x_hat": list(loc),
        "chebyshev": int(dist),
        "rho_Kvac": pearson(ds, pk),
        "newton": newt,
        "C_vac": c_vac,
        "C_mass": c_mass,
        "C_loc": c_loc,
        "C_newt_PRIMARY": c_newt,
        "all_gates": ok,
        "verdict": "KAPPA_TO_NEWTON" if ok else "PIPELINE_FAIL",
        "not_claimed": [
            "derived Poisson",
            "8pi G from kappa",
            "1/4G",
            "FGHMV",
            "de Sitter",
            "MODELS.md",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_38_from_kappa.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
