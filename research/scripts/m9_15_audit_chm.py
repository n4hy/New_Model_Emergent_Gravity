#!/usr/bin/env python3
"""CHM-shape audit on NN pairs. No solver import.

1d N=200 L=24: rho(K_nn, w_nn) < -0.70.
3d N=14 R=4: rho < -0.60 and R_shape < 0.50.

Writes ../data/m9_15_audit_chm.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
EPS = 1e-12


def pearson(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if den == 0.0 else float(np.dot(a, b) / den)


def rshape(k, w) -> float:
    k = np.asarray(k, float)
    w = np.asarray(w, float)
    mat = np.column_stack([w, np.ones(len(w))])
    coef, _, _, _ = np.linalg.lstsq(mat, k, rcond=None)
    den = float(np.linalg.norm(k - k.mean()))
    return float("nan") if den == 0.0 else float(np.linalg.norm(k - mat @ coef) / den)


def k_from_H(ham, sl):
    ev, vecs = np.linalg.eigh(ham)
    c = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    ca = c[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, EPS, 1.0 - EPS)
    return (u * np.log((1.0 - w) / w)) @ u.T


def audit_1d() -> dict:
    n, ell = 200, 24
    ham = np.zeros((n, n))
    for i in range(n - 1):
        ham[i, i + 1] = ham[i + 1, i] = -1.0
    mid = n // 2
    sl = np.arange(mid - ell // 2, mid + ell // 2)
    k = k_from_H(ham, sl)
    x = np.arange(ell) - (ell - 1) / 2.0
    w = (ell / 2.0) ** 2 - x**2
    knn = [k[i, i + 1] for i in range(ell - 1)]
    wnn = [0.5 * (w[i] + w[i + 1]) for i in range(ell - 1)]
    rho = pearson(knn, wnn)
    return {"rho": rho, "R_shape": rshape(knn, wnn), "C0": bool(rho < -0.70)}


def audit_3d() -> dict:
    n, radius = 14, 4

    def ix(x, y, z):
        return (x * n + y) * n + z

    vol = n**3
    ham = np.zeros((vol, vol))
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
    sl = []
    coords = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius * radius:
                    sl.append(ix(x, y, z))
                    coords.append((x, y, z))
    sl = np.array(sl)
    k = k_from_H(ham, sl)
    pos = {pt: i for i, pt in enumerate(coords)}
    w = [
        radius * radius - ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
        for x, y, z in coords
    ]
    knn, wnn = [], []
    for i, (x, y, z) in enumerate(coords):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            j = pos.get((x + d[0], y + d[1], z + d[2]))
            if j is not None:
                knn.append(k[i, j])
                wnn.append(0.5 * (w[i] + w[j]))
    rho = pearson(knn, wnn)
    rs = rshape(knn, wnn)
    rng = np.random.default_rng(15)
    rho_perm = pearson(knn, rng.permutation(wnn))
    return {
        "rho": rho,
        "R_shape": rs,
        "rho_perm": rho_perm,
        "C1": bool(rho < -0.60),
        "C2": bool(rs < 0.50),
        "C3": bool(abs(rho_perm) < 0.30),
    }


def main() -> int:
    d1 = audit_1d()
    d3 = audit_3d()
    ok = bool(d1["C0"] and d3["C1"] and d3["C2"] and d3["C3"])
    payload = {
        "task": "m9.15_audit_chm",
        "method": "NN kernel; 1d N=200 L=24; 3d N=14 R=4; no solver import",
        "one_d": d1,
        "three_d": d3,
        "verdicts": {"shape": "CONFIRMED" if ok else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_15_audit_chm.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
