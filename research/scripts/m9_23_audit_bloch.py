#!/usr/bin/env python3
"""Bloch vs CHM audit. No solver import.

1d N=180 L=24 all bonds. 3d N=14 R=5 ALL NN dimers.

Writes ../data/m9_23_audit_bloch.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12


def pearson(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if den == 0.0 else float(np.dot(a, b) / den)


def rshape(y, x) -> float:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    return float("nan") if den == 0.0 else float(np.linalg.norm(y - mat @ coef) / den)


def KC(ham, sl):
    ev, vecs = np.linalg.eigh(ham)
    corr = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    ca = corr[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, CLIP, 1.0 - CLIP)
    k = (u * np.log((1.0 - w) / w)) @ u.T
    return 0.5 * (k + k.T), ca


def audit_1d() -> dict:
    n, ell = 180, 24
    ham = np.zeros((n, n))
    for i in range(n - 1):
        ham[i, i + 1] = ham[i + 1, i] = -1.0
    mid = n // 2
    sl = np.arange(mid - ell // 2, mid + ell // 2)
    k, ca = KC(ham, sl)
    knn = [k[i, i + 1] for i in range(ell - 1)]
    rx = [2.0 * ca[i, i + 1] for i in range(ell - 1)]
    x = np.arange(ell) - (ell - 1) / 2.0
    w = [(ell / 2.0) ** 2 - (0.5 * (x[i] + x[i + 1])) ** 2 for i in range(ell - 1)]
    return {
        "n": ell - 1,
        "rho_bloch": pearson(knn, rx),
        "R_bloch": rshape(knn, rx),
        "R_chm": rshape(knn, w),
        "C0": bool(abs(pearson(knn, rx)) > 0.50),
        "C2": bool(rshape(knn, rx) < rshape(knn, w)),
    }


def audit_3d() -> dict:
    n, radius = 14, 5

    def ix(x, y, z):
        return (x * n + y) * n + z

    ham = np.zeros((n**3, n**3))
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = ix(x, y, z)
                if x + 1 < n:
                    ham[i, ix(x + 1, y, z)] = ham[ix(x + 1, y, z), i] = -1.0
                if y + 1 < n:
                    ham[i, ix(x, y + 1, z)] = ham[ix(x, y + 1, z), i] = -1.0
                if z + 1 < n:
                    ham[i, ix(x, y, z + 1)] = ham[ix(x, y, z + 1), i] = -1.0
    c = n // 2
    sl, coords = [], []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius * radius:
                    sl.append(ix(x, y, z))
                    coords.append((x, y, z))
    sl = np.array(sl)
    k, ca = KC(ham, sl)
    pos = {p: i for i, p in enumerate(coords)}
    knn, rx, wchm = [], [], []
    for i, (x, y, z) in enumerate(coords):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nbr = (x + d[0], y + d[1], z + d[2])
            j = pos.get(nbr)
            if j is None or j <= i:
                continue
            knn.append(k[i, j])
            rx.append(2.0 * ca[i, j])
            mx = 0.5 * (x + nbr[0] - 2 * c)
            my = 0.5 * (y + nbr[1] - 2 * c)
            mz = 0.5 * (z + nbr[2] - 2 * c)
            wchm.append(radius * radius - (mx * mx + my * my + mz * mz))
    rb, rc = rshape(knn, rx), rshape(knn, wchm)
    rho = pearson(knn, rx)
    return {
        "n": int(len(knn)),
        "n_ball": int(sl.size),
        "rho_bloch": rho,
        "R_bloch": rb,
        "R_chm": rc,
        "C1": bool(abs(rho) > 0.30),
        "C2": bool(rb < rc),
    }


def main() -> int:
    d1 = audit_1d()
    d3 = audit_3d()
    ok = bool(d1["C0"] and d3["C1"] and d3["C2"])
    payload = {
        "task": "m9.23_audit_bloch",
        "method": "ALL NN; 1d N=180 L=24; 3d N=14 R=5; no import",
        "one_d": d1,
        "three_d": d3,
        "verdicts": {"C2": "CONFIRMED" if ok else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_23_audit_bloch.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
