#!/usr/bin/env python3
"""M9.28: 3d balls, fixed-H occupation transfer.

Same construction as the 1d solver. Stagger is (−1)^{x+y+z}.
Scored only if C_vac passes.

PRE-REGISTERED:
  N=12, R=2 balls at every legal center (512).
  Packet at (6,6,6), σ=1.5. α=0.02 and 0.05.
  H fixed hop. e_i = ∑_j H_ij C_ij.
  C_vac  |ρ(δS, Tr(K_vac ΔC))| > 0.95.
  C0     max|δS| > 1e-6.
  C1     Pearson(δS(α), δS(2.5α)) > 0.95.
  C2     PRIMARY if C_vac. R_CHM < R_flat.
  C3     R_CHM < R_lin.
  C4     |ρ(δS, P_CHM)| > 0.60.

Writes ../data/m9_28_3d_state.json
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
SRC = (6, 6, 6)
SIGMA = 1.5
ALPHA1 = 0.02
ALPHA2 = 0.05


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


def occupation_transfer(ham: np.ndarray, n: int, src, sigma: float, alpha: float):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    u_occ = vecs[:, occ]
    u_un = vecs[:, ~occ]
    env = np.zeros((n**3,), dtype=float)
    stag = np.zeros((n**3,), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (sigma * sigma))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = u_occ @ (u_occ.T @ env)
    right = u_un @ (u_un.T @ stag)
    n_l = float(np.linalg.norm(left))
    n_r = float(np.linalg.norm(right))
    if n_l < 1e-14 or n_r < 1e-14:
        raise RuntimeError("packet vanished")
    left = left / n_l
    right = right / n_r
    c0 = u_occ @ u_occ.T
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return c0, 0.5 * (corr + corr.T), int(occ.sum()), n_l, n_r


def site_energy(ham: np.ndarray, corr: np.ndarray) -> np.ndarray:
    return np.sum(ham * corr, axis=1)


def peschel_s(corr: np.ndarray, sl: np.ndarray) -> float:
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr: np.ndarray, sl: np.ndarray) -> np.ndarray:
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def ball_sites(n: int, radius: int, cx: int, cy: int, cz: int) -> np.ndarray:
    sl = []
    r2 = radius * radius
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2:
                    sl.append(idx(x, y, z, n))
    return np.array(sl, dtype=int)


def ball_centers(n: int, radius: int):
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
    ham = hop_H(N)
    c0, c1, nocc, n_l, n_r = occupation_transfer(ham, N, SRC, SIGMA, ALPHA1)
    _, c2, _, _, _ = occupation_transfer(ham, N, SRC, SIGMA, ALPHA2)
    ev1 = np.linalg.eigvalsh(c1)
    de = site_energy(ham, c1) - site_energy(ham, c0)
    dc = c1 - c0
    centers = ball_centers(N, RADIUS)
    sls = [ball_sites(N, RADIUS, *c) for c in centers]
    ds1, ds2, pk0 = [], [], []
    for sl in sls:
        ds1.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        ds2.append(peschel_s(c2, sl) - peschel_s(c0, sl))
        pk0.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
    ds1 = np.asarray(ds1, dtype=float)
    ds2 = np.asarray(ds2, dtype=float)
    pk0 = np.asarray(pk0, dtype=float)
    pchm, plin, pflat = predictions(centers, de, N, RADIUS)
    r_chm = residual_ratio(ds1, pchm)
    r_lin = residual_ratio(ds1, plin)
    r_flat = residual_ratio(ds1, pflat)
    rho_k = pearson(ds1, pk0)
    rho_c = pearson(ds1, pchm)
    c_vac = bool(abs(rho_k) > 0.95)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6)
    c1g = bool(pearson(ds1, ds2) > 0.95)
    c2g = bool(r_chm < r_flat)
    c3g = bool(r_chm < r_lin)
    c4g = bool(abs(rho_c) > 0.60)
    ok = bool(c_vac and c0g and c1g and c2g and c4g)
    payload = {
        "task": "m9.28_3d_state",
        "H_fixed": True,
        "construction": "occupation_transfer",
        "n_balls": int(len(centers)),
        "n_occ": int(nocc),
        "packet_norms": [n_l, n_r],
        "c_eig_min": float(np.min(ev1)),
        "c_eig_max": float(np.max(ev1)),
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "rho_eps": pearson(ds1, ds2),
        "rho_CHM": rho_c,
        "rho_flat": pearson(ds1, pflat),
        "rho_linear_w": pearson(ds1, plin),
        "R_CHM": r_chm,
        "R_lin": r_lin,
        "R_flat": r_flat,
        "rho_Kvac": rho_k,
        "C_vac": c_vac,
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C2_PRIMARY_chm_beats_flat": c2g,
        "C3_chm_beats_linear": c3g,
        "C4_tracks": c4g,
        "all_gates": ok,
        "verdict": "3D_FIXEDH_CHM_WINS" if ok else (
            "3D_FIXEDH_INSTRUMENT_REJECT" if not c_vac else "3D_FIXEDH_FLAT"
        ),
        "not_claimed": ["Einstein equation", "8pi G", "de Sitter"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_28_3d_state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
