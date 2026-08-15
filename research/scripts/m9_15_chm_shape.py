#!/usr/bin/env python3
"""M9.15: CHM shape of the modular hopping kernel.

CHM / Bisognano-Wichmann: K = 2 pi ∫ w T_00, with w = R^2 - r^2
(ball) or w = (L/2)^2 - x^2 (interval). T_00 contains the hop, so
the test is on nearest-neighbour K_ij, not on the diagonal.

First instrument (diagonal vs w) failed C0 on the 1d theorem and
was rejected. This file is the CHM-correct observable.

PRE-REGISTERED:
  C0  INSTRUMENT. 1d massless interval, NN pairs:
        rho(K_ij, (w_i+w_j)/2) < -0.70.
  C1  3+1D ball R=5, m=0, NN pairs: rho < -0.60.
  C2  PRIMARY. Residual of K_ij ~ a w_nn + b
        R_shape < 0.50 at m=0, R=5.
  C3  Mutation: a fixed permutation of w_nn (seed 15) has
        |rho_perm| < 0.30 at m=0. (r^2 is affine to -w and
        cannot be used as a mutation.)
  C4  C1 still holds at m R = 0.5.

Not claimed: eta=1/4G, foam, de Sitter, FGHMV.

Writes ../data/m9_15_chm_shape.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

EPS = 1e-12
N1 = 240
L1 = 32
N3 = 16
R3 = 5
M_UV = 0.10


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def residual_ratio(k: np.ndarray, w: np.ndarray) -> float:
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    mat = np.column_stack([w, np.ones(len(w))])
    coef, _, _, _ = np.linalg.lstsq(mat, k, rcond=None)
    pred = mat @ coef
    den = float(np.linalg.norm(k - k.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(k - pred) / den)


def modular_from_H(ham: np.ndarray, sl: np.ndarray) -> np.ndarray:
    ev, vecs = np.linalg.eigh(ham)
    corr = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    ca = corr[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, EPS, 1.0 - EPS)
    k = (u * np.log((1.0 - w) / w)) @ u.T
    return 0.5 * (k + k.T)


def one_d() -> dict:
    ham = np.zeros((N1, N1), dtype=float)
    for i in range(N1 - 1):
        ham[i, i + 1] = ham[i + 1, i] = -1.0
    mid = N1 // 2
    sl = np.arange(mid - L1 // 2, mid + L1 // 2)
    k = modular_from_H(ham, sl)
    x = np.arange(L1) - (L1 - 1) / 2.0
    w = (L1 / 2.0) ** 2 - x**2
    knn = np.array([k[i, i + 1] for i in range(L1 - 1)], dtype=float)
    wnn = np.array([0.5 * (w[i] + w[i + 1]) for i in range(L1 - 1)])
    r2nn = np.array([0.5 * (x[i] ** 2 + x[i + 1] ** 2) for i in range(L1 - 1)])
    rho = pearson(knn, wnn)
    return {
        "N": N1,
        "L": L1,
        "n_nn": int(knn.size),
        "rho_chm": rho,
        "R_shape": residual_ratio(knn, wnn),
        "rho_r2": pearson(knn, r2nn),
        "C0": bool(rho < -0.70),
    }


def idx(x: int, y: int, z: int, n: int) -> int:
    return (x * n + y) * n + z


def staggered_H_3d(n: int, mass: float) -> np.ndarray:
    vol = n**3
    ham = np.zeros((vol, vol), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                ham[i, i] = mass * (1.0 if (x + y + z) % 2 == 0 else -1.0)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham[i, j] = ham[j, i] = -1.0
    return ham


def three_d(mass: float) -> dict:
    n, radius = N3, R3
    sl_list = []
    coords = []
    c = n // 2
    r2max = radius * radius
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= r2max:
                    sl_list.append(idx(x, y, z, n))
                    coords.append((x, y, z))
    sl = np.array(sl_list)
    k = modular_from_H(staggered_H_3d(n, mass), sl)
    pos = {pt: i for i, pt in enumerate(coords)}
    w = np.array(
        [r2max - ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) for x, y, z in coords],
        dtype=float,
    )
    knn = []
    wnn = []
    for i, (x, y, z) in enumerate(coords):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            j = pos.get((x + d[0], y + d[1], z + d[2]))
            if j is not None:
                knn.append(k[i, j])
                wnn.append(0.5 * (w[i] + w[j]))
    knn = np.array(knn)
    wnn = np.array(wnn)
    rho = pearson(knn, wnn)
    rng = np.random.default_rng(15)
    w_perm = rng.permutation(wnn)
    rho_perm = pearson(knn, w_perm)
    return {
        "m": mass,
        "mR": mass * radius,
        "n_ball": int(sl.size),
        "n_nn": int(knn.size),
        "rho_chm": rho,
        "R_shape": residual_ratio(knn, wnn),
        "rho_perm": rho_perm,
    }


def main() -> int:
    cal = one_d()
    row0 = three_d(0.0)
    row_uv = three_d(M_UV)
    c0 = bool(cal["C0"])
    c1 = bool(row0["rho_chm"] < -0.60)
    c2 = bool(row0["R_shape"] < 0.50)
    c3 = bool(abs(row0["rho_perm"]) < 0.30)
    c4 = bool(row_uv["rho_chm"] < -0.60)
    ok = bool(c0 and c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.15_chm_shape",
        "observable": "NN K_ij vs CHM envelope (w_i+w_j)/2",
        "one_d_calibration": cal,
        "ball_m0": row0,
        "ball_uv": row_uv,
        "C0_instrument": c0,
        "C1_rho": c1,
        "C2_PRIMARY_Rshape": c2,
        "C3_mutation": c3,
        "C4_uv_stable": c4,
        "all_gates": ok,
        "verdict": "CHM_SHAPE_PASS" if ok else "CHM_SHAPE_FAIL",
        "not_claimed": [
            "eta = 1/4G",
            "mean-zero foam",
            "de Sitter",
            "FGHMV",
            "a continuum theorem",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_15_chm_shape.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
