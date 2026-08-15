#!/usr/bin/env python3
"""M9.52 audit. N=10, R=3, src (4,5,5), σ=0.9, α ∈ {0.01, 0.06}.

Own 64 centres. Tries to REFUTE a stable winner.

Writes ../data/m9_52_audit_pred.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG = 10, 3, 0.9
SRC = (4, 5, 5)
ALPHAS = (0.01, 0.06)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


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


def main() -> int:
    ham = np.zeros((N**3, N**3))
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
    env = np.zeros(N**3)
    stag = np.zeros(N**3)
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
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    c0 = uo @ uo.T
    dC = np.outer(right, right) - np.outer(left, left)
    de1 = np.sum(ham * dC, axis=1)
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    r2 = RADIUS * RADIUS
    slices, chm_w = [], []
    for cx, cy, cz in centers:
        sl, w = [], []
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2:
                        sl.append(idx(x, y, z))
                        w.append(r2 - rr)
        slices.append(np.array(sl, dtype=int))
        chm_w.append(np.asarray(w, float))
    s0 = [peschel_s(c0, sl) for sl in slices]
    tk1, pf1, pc1 = [], [], []
    for sl, w in zip(slices, chm_w):
        kv = peschel_k(c0, sl)
        tk1.append(float(np.sum(kv * dC[np.ix_(sl, sl)])))
        pf1.append(float(np.sum(de1[sl])))
        pc1.append(float(np.sum(w * de1[sl])))
    tk1, pf1, pc1 = map(np.asarray, (tk1, pf1, pc1))
    winners = []
    rows = []
    for alpha in ALPHAS:
        c1 = 0.5 * ((c0 + alpha * dC) + (c0 + alpha * dC).T)
        ds = np.array([peschel_s(c1, sl) - s0[i] for i, sl in enumerate(slices)])
        rhos = {
            "P_flat": pearson(ds, alpha * pf1),
            "P_CHM": pearson(ds, alpha * pc1),
            "T_K": pearson(ds, alpha * tk1),
        }
        win = max(rhos, key=lambda n: abs(rhos[n]))
        winners.append(win)
        rows.append({"alpha": alpha, "rho": rhos, "winner": win})
    payload = {
        "task": "m9.52_audit_pred",
        "n_balls": len(centers),
        "rows": rows,
        "winners": winners,
        "C_flip": bool(len(set(winners)) > 1),
        "verdicts": {"C_flip": "CONFIRMED" if len(set(winners)) > 1 else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_52_audit_pred.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
