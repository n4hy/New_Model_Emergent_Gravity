#!/usr/bin/env python3
"""M9.26: point source so CHM vs flat can separate.

Paper 34's Gaussian made all kernels look the same. A single-site
potential makes P_flat ~ 1_{source in ball} while P_CHM ~ w(source)
(large if the source is at the ball center, small at the rim).

PRE-REGISTERED:
  N=12, R=2 balls at every legal center (512).
  V = ε at site (6,6,6) only. ε=0.05 and 0.10.
  E<0; half-fill if n_occ flips.
  C0  max|δS| > 1e-6 (absolute; Paper 34's S-ratio was a bad lock).
  C1  Pearson(δS(ε), δS(2ε)) > 0.95.
  C2  PRIMARY. R_shape(δS, P_CHM) < R_shape(δS, P_flat).
  C3  R_shape(δS, P_CHM) < R_shape(δS, P_lin).
  C4  |ρ(δS, P_CHM)| > 0.60.

Not claimed: Einstein, 8πG, dS, SM.

Writes ../data/m9_26_point_source.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
RADIUS = 2
EPS1 = 0.05
EPS2 = 0.10
SRC = (6, 6, 6)


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H(n: int) -> np.ndarray:
    vol = n**3
    ham = np.zeros((vol, vol), dtype=float)
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


def point_V(n: int, eps: float, site: tuple[int, int, int]) -> np.ndarray:
    pot = np.zeros((n**3,), dtype=float)
    pot[idx(*site, n)] = eps
    return pot


def occupy(ham: np.ndarray, half: bool) -> tuple[np.ndarray, int]:
    ev, vecs = np.linalg.eigh(ham)
    if half:
        nocc = ham.shape[0] // 2
        filled = np.zeros(ev.shape, dtype=bool)
        filled[:nocc] = True
    else:
        filled = ev < 0.0
        nocc = int(filled.sum())
    return vecs[:, filled] @ vecs[:, filled].T, nocc


def site_energy(ham: np.ndarray, corr: np.ndarray) -> np.ndarray:
    return np.sum(ham * corr, axis=1)


def peschel_s(corr: np.ndarray, sl: np.ndarray) -> float:
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def ball_sites(n: int, radius: int, cx: int, cy: int, cz: int) -> np.ndarray:
    sl = []
    r2 = radius * radius
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2:
                    sl.append(idx(x, y, z, n))
    return np.array(sl, dtype=int)


def ball_centers(n: int, radius: int) -> list[tuple[int, int, int]]:
    lo, hi = radius, n - radius
    return [(x, y, z) for x in range(lo, hi) for y in range(lo, hi) for z in range(lo, hi)]


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def residual_ratio(y, x) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(y - mat @ coef) / den)


def predictions(centers, de, n, radius):
    pchm, plin, pflat = [], [], []
    r2max = radius * radius
    for cx, cy, cz in centers:
        s_c = s_l = s_f = 0.0
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr > r2max:
                        continue
                    e = de[idx(x, y, z, n)]
                    s_c += (r2max - rr) * e
                    s_l += (radius - np.sqrt(rr)) * e
                    s_f += e
        pchm.append(s_c)
        plin.append(s_l)
        pflat.append(s_f)
    return np.array(pchm), np.array(plin), np.array(pflat)


def main() -> int:
    ham0 = hop_H(N)
    c0, n0 = occupy(ham0, half=False)
    pot = point_V(N, 1.0, SRC)
    ham1 = ham0.copy()
    np.fill_diagonal(ham1, ham1.diagonal() + EPS1 * pot)
    ham2 = ham0.copy()
    np.fill_diagonal(ham2, ham2.diagonal() + EPS2 * pot)
    c1, n1 = occupy(ham1, half=False)
    c2, n2 = occupy(ham2, half=False)
    half = False
    if n1 != n0 or n2 != n0:
        half = True
        c0, n0 = occupy(ham0, half=True)
        c1, n1 = occupy(ham1, half=True)
        c2, n2 = occupy(ham2, half=True)
    e0 = site_energy(ham0, c0)
    e1 = site_energy(ham1, c1)
    de = e1 - e0
    centers = ball_centers(N, RADIUS)
    sls = [ball_sites(N, RADIUS, *c) for c in centers]
    s0 = np.array([peschel_s(c0, sl) for sl in sls])
    s1 = np.array([peschel_s(c1, sl) for sl in sls])
    s2 = np.array([peschel_s(c2, sl) for sl in sls])
    ds1 = s1 - s0
    ds2 = s2 - s0
    pchm, plin, pflat = predictions(centers, de, N, RADIUS)
    rho_lin = pearson(ds1, ds2)
    r_chm = residual_ratio(ds1, pchm)
    r_lin = residual_ratio(ds1, plin)
    r_flat = residual_ratio(ds1, pflat)
    rho_chm = pearson(ds1, pchm)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6)
    c1g = bool(rho_lin > 0.95)
    c2g = bool(r_chm < r_flat)
    c3g = bool(r_chm < r_lin)
    c4g = bool(abs(rho_chm) > 0.60)
    ok = bool(c0g and c1g and c2g and c4g)
    payload = {
        "task": "m9.26_point_source",
        "source": "single site (6,6,6)",
        "n_balls": int(len(centers)),
        "half_fill": half,
        "n_occ": [int(n0), int(n1), int(n2)],
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "mean_S0": float(np.mean(s0)),
        "rho_dS_eps": rho_lin,
        "rho_CHM": rho_chm,
        "R_CHM": r_chm,
        "R_lin": r_lin,
        "R_flat": r_flat,
        "rho_flat": pearson(ds1, pflat),
        "rho_linear_w": pearson(ds1, plin),
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C2_PRIMARY_chm_beats_flat": c2g,
        "C3_chm_beats_linear": c3g,
        "C4_tracks": c4g,
        "all_gates": ok,
        "verdict": "POINT_CHM_WINS" if ok else "POINT_KERNEL_NOT_CHM",
        "not_claimed": [
            "Einstein equation",
            "8pi G",
            "FGHMV in AdS",
            "de Sitter",
            "Standard Model",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_26_point_source.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
