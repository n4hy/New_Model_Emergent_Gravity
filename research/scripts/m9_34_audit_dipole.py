#!/usr/bin/env python3
"""M9.34 audit. N=10, sources (4,5,5) and (6,5,5), σ=0.9, α=0.03.

Two same-sign PH packets. Tries to REFUTE C2b (CHM beats
flat on balls that contain both).

Writes ../data/m9_34_audit_dipole.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, ALPHA = 10, 2, 0.9, 0.03
A, B = (4, 5, 5), (6, 5, 5)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


def rshape(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    if len(y) < 3:
        return float("nan")
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

    def raw(src):
        env = np.zeros(vol)
        stag = np.zeros(vol)
        sx, sy, sz = src
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                    env[i] = np.exp(-0.5 * rr / (SIG * SIG))
                    stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
        return uo @ (uo.T @ env), uu @ (uu.T @ stag)

    def onorm(v1, v2):
        e1 = v1 / np.linalg.norm(v1)
        v2 = v2 - e1 * np.dot(e1, v2)
        return e1, v2 / np.linalg.norm(v2)

    la, ra = raw(A)
    lb, rb = raw(B)
    la, lb = onorm(la, lb)
    ra, rb = onorm(ra, rb)
    c0 = uo @ uo.T
    c1 = c0 + ALPHA * (np.outer(ra, ra) - np.outer(la, la) + np.outer(rb, rb) - np.outer(lb, lb))
    c1 = 0.5 * (c1 + c1.T)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    r2max = RADIUS * RADIUS

    def S(c, sl):
        z = np.clip(np.linalg.eigvalsh(c[np.ix_(sl, sl)]), CLIP, 1 - CLIP)
        return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

    def inside(src, c):
        return sum((src[k] - c[k]) ** 2 for k in range(3)) <= r2max

    ds, pc, pf, both = [], [], [], []
    for cx in range(RADIUS, N - RADIUS):
        for cy in range(RADIUS, N - RADIUS):
            for cz in range(RADIUS, N - RADIUS):
                sl, s_c, s_f = [], 0.0, 0.0
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
                ds.append(S(c1, sl) - S(c0, sl))
                pc.append(s_c)
                pf.append(s_f)
                both.append(inside(A, (cx, cy, cz)) and inside(B, (cx, cy, cz)))
    ds, pc, pf = map(np.asarray, (ds, pc, pf))
    both = np.asarray(both, bool)
    rc, rf = rshape(ds, pc), rshape(ds, pf)
    n_b = int(both.sum())
    rcb = rshape(ds[both], pc[both]) if n_b >= 10 else float("nan")
    rfb = rshape(ds[both], pf[both]) if n_b >= 10 else float("nan")
    c2 = bool(rc < rf)
    c2b = bool(n_b >= 10 and rcb < rfb)
    payload = {
        "task": "m9.34_audit_pair",
        "n": int(len(ds)),
        "n_both": n_b,
        "rho_CHM": pearson(ds, pc),
        "rho_flat": pearson(ds, pf),
        "R_CHM": rc,
        "R_flat": rf,
        "R_CHM_both": rcb,
        "R_flat_both": rfb,
        "C2": c2,
        "C2b": c2b,
        "verdicts": {
            "C2": "CONFIRMED" if c2 else "REFUTED",
            "C2b": "CONFIRMED" if c2b else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_34_audit_dipole.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c2 and c2b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
