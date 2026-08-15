#!/usr/bin/env python3
"""M9.25: is δS of balls a linear functional of local energy with CHM kernel?

Closest honest step toward 'entanglement ⇒ linearized gravity'.
One vacuum, one weak Gaussian potential, MANY balls (all centers
that fit). Compare δS to ∑_ball w δe with w = R²-r² (CHM),
w = R-r (linear), w = 1 (flat).

Site energy e_i = ∑_j H_ij C_ij so ∑_i e_i = Tr(H C).

PRE-REGISTERED:
  N=12, R=2 balls at every center with a 2-site margin.
  V_i = ε exp(-|r-c|²/(2σ²)), σ=2, ε=0.05 and 0.10.
  Occupy E<0; if n_occ flips, fall back to V/2 and record it.
  C0  max|δS|/mean S_vac > 1e-4 (a signal exists).
  C1  Pearson(δS(ε), δS(2ε)) > 0.95 (linear regime).
  C2  PRIMARY. R_shape(δS, P_CHM) < R_shape(δS, P_flat) at ε=0.05.
  C3  R_shape(δS, P_CHM) < R_shape(δS, P_lin) at ε=0.05.
  C4  |ρ(δS, P_CHM)| > 0.60 at ε=0.05 (actually tracks).

Not claimed: Einstein equation, 8πG, Planck, dS, SM, FGHMV in AdS.

Writes ../data/m9_25_linear_functional.json
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
SIGMA = 2.0
EPS1 = 0.05
EPS2 = 0.10


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


def gaussian_V(n: int, eps: float, sigma: float) -> np.ndarray:
    c = n / 2.0 - 0.5
    pot = np.zeros((n**3,), dtype=float)
    s2 = 2.0 * sigma * sigma
    for x in range(n):
        for y in range(n):
            for z in range(n):
                r2 = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
                pot[idx(x, y, z, n)] = eps * np.exp(-r2 / s2)
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
    pot = gaussian_V(N, 1.0, SIGMA)
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
    c0g = bool(float(np.max(np.abs(ds1))) / max(float(np.mean(s0)), 1e-12) > 1e-4)
    c1g = bool(rho_lin > 0.95)
    c2g = bool(r_chm < r_flat)
    c3g = bool(r_chm < r_lin)
    c4g = bool(abs(rho_chm) > 0.60)
    ok = bool(c0g and c1g and c2g and c4g)
    payload = {
        "task": "m9.25_linear_functional",
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
        "verdict": "LINEAR_CHM_FUNCTIONAL" if ok else "NOT_CHM_LINEAR_FUNCTIONAL",
        "not_claimed": [
            "Einstein equation",
            "8pi G",
            "FGHMV in AdS",
            "de Sitter",
            "Standard Model",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_25_linear_functional.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
