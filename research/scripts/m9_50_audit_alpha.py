#!/usr/bin/env python3
"""M9.50 audit. N=10, src (4,5,5), σ=0.9, R=3.
α ∈ {0.01, 0.03, 0.06}. Tries to REFUTE C_lin.

Writes ../data/m9_50_audit_alpha.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SRC, SIG = 10, (4, 5, 5), 0.9
ALPHAS = (0.01, 0.03, 0.06)
RADIUS = 3


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def h_bin(a):
    return float(-a * np.log(a) - (1.0 - a) * np.log(1.0 - a))


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
    sl = np.array(
        [
            idx(x, y, z)
            for x in range(N)
            for y in range(N)
            for z in range(N)
            if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= RADIUS * RADIUS
        ],
        dtype=int,
    )
    s0 = peschel_s(c0, sl)
    e0 = np.sum(ham * c0, axis=1)
    kappas, rsgs = [], []
    for alpha in ALPHAS:
        c1 = 0.5 * (
            (c0 + alpha * (np.outer(right, right) - np.outer(left, left)))
            + (c0 + alpha * (np.outer(right, right) - np.outer(left, left))).T
        )
        de = np.sum(ham * c1, axis=1) - e0
        ds = peschel_s(c1, sl) - s0
        p = float(np.sum(de[sl]))
        kappas.append(ds / p)
        rsgs.append(2.0 * h_bin(alpha) / p)
    kappas, rsgs = np.asarray(kappas), np.asarray(rsgs)
    rel_k = float((kappas.max() - kappas.min()) / np.median(kappas))
    rel_r = float((rsgs.max() - rsgs.min()) / np.median(rsgs))
    c_lin = bool(rel_k < 0.10)
    c_sg = bool(rel_r > 0.20)
    payload = {
        "task": "m9.50_audit_alpha",
        "kappas": kappas.tolist(),
        "r_sg": rsgs.tolist(),
        "rel_kappa": rel_k,
        "rel_r_sg": rel_r,
        "C_lin": c_lin,
        "C_sg": c_sg,
        "verdicts": {
            "C_lin": "CONFIRMED" if c_lin else "REFUTED",
            "C_sg": "CONFIRMED" if c_sg else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_50_audit_alpha.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_lin and c_sg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
