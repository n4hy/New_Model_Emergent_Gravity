#!/usr/bin/env python3
"""M9.37 audit. N=10, R=3, source (4,5,5), σ=0.9, α=0.03, seed 91.

Own split, own intersection locate. Tries to REFUTE C_mass and C_loc.

Writes ../data/m9_37_audit_weigh.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, ALPHA, SEED, HIGH = 10, 3, 0.9, 0.03, 91, 0.70
SRC = (4, 5, 5)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


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
    c1 = c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))
    c1 = 0.5 * (c1 + c1.T)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    m_true = float(np.sum(de))
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
        inside.append(
            (cx - SRC[0]) ** 2 + (cy - SRC[1]) ** 2 + (cz - SRC[2]) ** 2 <= r2max
        )
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
        mv = np.where(survive, votes, -1)
        loc = np.unravel_index(int(np.argmax(mv)), votes.shape)
    else:
        loc = np.unravel_index(int(np.argmax(votes)), votes.shape)
    loc = (int(loc[0]), int(loc[1]), int(loc[2]))
    dist = max(abs(loc[k] - SRC[k]) for k in range(3))
    c_mass = bool(mass_rel is not None and mass_rel < 0.10)
    c_loc = bool(dist == 0)
    payload = {
        "task": "m9.37_audit_weigh",
        "kappa": kappa,
        "M_true_global": m_true,
        "M_enc": m_enc,
        "M_hat": m_hat,
        "mass_rel": mass_rel,
        "x_true": list(SRC),
        "x_hat": list(loc),
        "chebyshev": int(dist),
        "C_mass": c_mass,
        "C_loc": c_loc,
        "verdicts": {
            "C_mass": "CONFIRMED" if c_mass else "REFUTED",
            "C_loc": "CONFIRMED" if c_loc else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_37_audit_weigh.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_mass and c_loc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
