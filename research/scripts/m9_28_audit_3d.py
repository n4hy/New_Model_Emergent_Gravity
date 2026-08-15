#!/usr/bin/env python3
"""3d fixed-H occupation-transfer audit. N=10, R=2, src=(4,4,4), σ=1.2, α=0.03.

No import of the solver. Tries to REFUTE C_vac and C2.

Writes ../data/m9_28_audit_3d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, ALPHA = 10, 2, 1.2, 0.03
SRC = (4, 4, 4)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


def rshape(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    m = np.column_stack([x, np.ones(len(x))])
    c, _, _, _ = np.linalg.lstsq(m, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    return float("nan") if den == 0 else float(np.linalg.norm(y - m @ c) / den)


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
    dc = c1 - c0
    lo, hi = RADIUS, N - RADIUS
    centers = [(x, y, z) for x in range(lo, hi) for y in range(lo, hi) for z in range(lo, hi)]
    r2max = RADIUS * RADIUS
    ds, pc, pf, pk = [], [], [], []
    for cx, cy, cz in centers:
        sl = []
        s_c = s_f = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        i = idx(x, y, z)
                        sl.append(i)
                        s_c += (r2max - rr) * de[i]
                        s_f += de[i]
        sl = np.array(sl, dtype=int)
        b0 = c0[np.ix_(sl, sl)]
        b1 = c1[np.ix_(sl, sl)]
        z0 = np.clip(np.linalg.eigvalsh(b0), CLIP, 1 - CLIP)
        z1 = np.clip(np.linalg.eigvalsh(b1), CLIP, 1 - CLIP)
        ds.append(float(-np.sum(z1 * np.log(z1) + (1 - z1) * np.log(1 - z1))
                        + np.sum(z0 * np.log(z0) + (1 - z0) * np.log(1 - z0))))
        evk, uk = np.linalg.eigh(b0)
        evk = np.clip(evk, CLIP, 1 - CLIP)
        k0 = (uk * np.log((1 - evk) / evk)) @ uk.T
        pk.append(float(np.sum(k0 * dc[np.ix_(sl, sl)])))
        pc.append(s_c)
        pf.append(s_f)
    ds = np.asarray(ds, float)
    rc, rf = rshape(ds, pc), rshape(ds, pf)
    rho_k = pearson(ds, pk)
    c_vac = bool(abs(rho_k) > 0.95)
    c2 = bool(rc < rf)
    payload = {
        "task": "m9.28_audit_3d",
        "n": int(len(ds)),
        "rho_CHM": pearson(ds, pc),
        "rho_flat": pearson(ds, pf),
        "R_CHM": rc,
        "R_flat": rf,
        "rho_Kvac": rho_k,
        "C_vac": c_vac,
        "C2": c2,
        "verdicts": {
            "C_vac": "CONFIRMED" if c_vac else "REFUTED",
            "C2": "CONFIRMED" if c2 else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_28_audit_3d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_vac and c2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
