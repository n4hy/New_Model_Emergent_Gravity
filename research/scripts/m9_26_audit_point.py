#!/usr/bin/env python3
"""Point-source audit. N=10, R=2, ε=0.08 at site (3,3,3). No import.

Writes ../data/m9_26_audit_point.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, EPS = 10, 2, 0.08
SRC = (3, 3, 3)


def ix(x, y, z, n=N):
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
                i = ix(x, y, z)
                if x + 1 < N:
                    ham[i, ix(x + 1, y, z)] = ham[ix(x + 1, y, z), i] = -1.0
                if y + 1 < N:
                    ham[i, ix(x, y + 1, z)] = ham[ix(x, y + 1, z), i] = -1.0
                if z + 1 < N:
                    ham[i, ix(x, y, z + 1)] = ham[ix(x, y, z + 1), i] = -1.0
    pot = np.zeros(vol)
    pot[ix(*SRC)] = EPS
    ham1 = ham.copy()
    np.fill_diagonal(ham1, ham1.diagonal() + pot)

    def C_of(h):
        ev, vecs = np.linalg.eigh(h)
        return vecs[:, ev < 0] @ vecs[:, ev < 0].T

    c0, c1 = C_of(ham), C_of(ham1)
    de = np.sum(ham1 * c1, axis=1) - np.sum(ham * c0, axis=1)
    lo, hi = RADIUS, N - RADIUS
    centers = [(x, y, z) for x in range(lo, hi) for y in range(lo, hi) for z in range(lo, hi)]
    r2m = RADIUS * RADIUS
    ds, pchm, pflat = [], [], []
    for cx, cy, cz in centers:
        sl = []
        sc = sf = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr > r2m:
                        continue
                    sl.append(ix(x, y, z))
                    sc += (r2m - rr) * de[ix(x, y, z)]
                    sf += de[ix(x, y, z)]
        sl = np.array(sl)
        def S(c):
            w = np.clip(np.linalg.eigvalsh(c[np.ix_(sl, sl)]), CLIP, 1 - CLIP)
            return float(-np.sum(w * np.log(w) + (1 - w) * np.log(1 - w)))
        ds.append(S(c1) - S(c0))
        pchm.append(sc)
        pflat.append(sf)
    rc, rf = rshape(ds, pchm), rshape(ds, pflat)
    rho = pearson(ds, pchm)
    c2 = bool(rc < rf)
    c4 = bool(abs(rho) > 0.60)
    payload = {
        "task": "m9.26_audit_point",
        "n_balls": int(len(centers)),
        "rho_CHM": rho,
        "R_CHM": rc,
        "R_flat": rf,
        "C2": c2,
        "C4": c4,
        "verdicts": {"C2": "CONFIRMED" if c2 else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_26_audit_point.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
