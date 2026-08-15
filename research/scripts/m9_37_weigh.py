#!/usr/bin/env python3
"""M9.37: use κ. Weigh and locate the mass from δS.

Paper 46: δS = κ M_enc well inside, κ universal to ~2%.
This run treats κ as an instrument.

PRE-REGISTERED:
  N=12, R=3, 216 balls. One packet at (6,6,6), σ=1.0, α=0.02.
  κ from a random even/odd split of well-inside balls
  (seed 37): κ = median(δS/P_flat) on the even subset.
  C_vac   |ρ(δS, Tr(K_vac ΔC))| > 0.95
  C_mass  PRIMARY. |M_hat/M_enc − 1| < 0.10
          M_hat = median(δS/κ) on the odd well-inside subset
          M_enc = median(P_flat) on that same odd subset
          (κ sees enclosed energy, not leaked tails of ∑δe).
  C_loc   reconstructed site is the true source
          (intersection of balls with δS > 0.70 max δS;
          among surviving sites, pick the one in the most
          high-δS balls). Distance 0 required.
  C_pred  |ρ(δS, κ P_flat)| > 0.95 on all well-inside balls
  Inherited diagnostic (not a gate): DST Poisson of a
  one-site blob of mass M_hat at x_hat. Not claimed as
  entanglement gravity.

Not claimed: 8πG, 1/4G, Einstein, de Sitter, MODELS.md.

Writes ../data/m9_37_weigh.json
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
SEED = 37
HIGH = 0.70


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


def poisson_point(n, site, mass, g=1.0):
    """Dirichlet DST Poisson of a one-site blob. Inherited diagnostic."""
    rhs = np.zeros((n, n, n), dtype=float)
    sx, sy, sz = site
    rhs[sx, sy, sz] = 4.0 * np.pi * g * mass
    h = 1.0
    m = n - 2
    fhat = rhs[1:-1, 1:-1, 1:-1]
    for ax in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=ax)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for ax in (0, 1, 2):
        phat = dst(phat, type=1, axis=ax)
    phi = np.zeros((n, n, n), dtype=float)
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    return phi


def main() -> int:
    ham = hop_H(N)
    c0, c1 = occupation_transfer(ham, N, SRC, SIGMA, ALPHA)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    m_true = float(np.sum(de))
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds, pflat, pk, inside, sls = [], [], [], [], []
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
        sls.append(sl)
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pk.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
        s_f = float(np.sum(de[sl]))
        pflat.append(s_f)
        inside.append(in_ball(SRC, (cx, cy, cz), RADIUS))
    ds, pflat, pk = map(np.asarray, (ds, pflat, pk))
    inside = np.asarray(inside, bool)
    well = inside & (np.abs(pflat) > 1e-6)
    rng = np.random.default_rng(SEED)
    well_idx = np.flatnonzero(well)
    rng.shuffle(well_idx)
    half = len(well_idx) // 2
    even, odd = well_idx[:half], well_idx[half:]
    kappa = float(np.median(ds[even] / pflat[even]))
    m_hat = float(np.median(ds[odd] / kappa))
    m_enc = float(np.median(pflat[odd]))
    mass_rel = abs(m_hat / m_enc - 1.0) if m_enc != 0.0 else None
    leak = abs(m_enc / m_true - 1.0) if m_true != 0.0 else None
    # locate
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
        if first:
            survive = mask
            first = False
        else:
            survive &= mask
    if not first and np.any(survive):
        masked_votes = np.where(survive, votes, -1)
        loc = np.unravel_index(int(np.argmax(masked_votes)), votes.shape)
    else:
        loc = np.unravel_index(int(np.argmax(votes)), votes.shape)
    loc = (int(loc[0]), int(loc[1]), int(loc[2]))
    dist = max(abs(loc[k] - SRC[k]) for k in range(3))
    # inherited Poisson diagnostic at reconstructed site
    phi = poisson_point(N, loc, m_hat)
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / 2.0
    probes = []
    for d in (3, 4, 5):
        x = loc[0] + d
        if 1 <= x <= N - 2:
            ar = float(ax[x, loc[1], loc[2]])
            probes.append({"d": d, "a": ar, "attractive": bool(ar < 0.0)})
    c_vac = bool(abs(pearson(ds, pk)) > 0.95)
    c_mass = bool(mass_rel is not None and mass_rel < 0.10)
    c_loc = bool(dist == 0)
    c_pred = bool(abs(pearson(ds[well], kappa * pflat[well])) > 0.95)
    ok = bool(c_vac and c_mass and c_loc and c_pred)
    payload = {
        "task": "m9.37_weigh",
        "n_balls": int(len(centers)),
        "n_well": int(well.sum()),
        "n_even": int(len(even)),
        "n_odd": int(len(odd)),
        "kappa": kappa,
        "M_true_global": m_true,
        "M_enc": m_enc,
        "M_hat": m_hat,
        "mass_rel": mass_rel,
        "enc_vs_global": leak,
        "x_true": list(SRC),
        "x_hat": list(loc),
        "chebyshev": int(dist),
        "n_high": int(high.sum()),
        "rho_Kvac": pearson(ds, pk),
        "rho_pred": pearson(ds[well], kappa * pflat[well]),
        "inherited_probes": probes,
        "C_vac": c_vac,
        "C_mass_PRIMARY": c_mass,
        "C_loc": c_loc,
        "C_pred": c_pred,
        "all_gates": ok,
        "verdict": "WEIGHED_AND_LOCATED" if ok else "WEIGH_FAIL",
        "not_claimed": ["8pi G", "1/4G", "Einstein equation", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_37_weigh.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
