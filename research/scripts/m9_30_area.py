#!/usr/bin/env python3
"""M9.30: the geometric half. δS versus lattice area.

Fixed-H matter (Papers 37–38) cannot be gravity: hop-length
area does not change. This run rescales hops by a conformal
bump Φ (a metric, not a potential in H_ii) and asks whether
δS tracks δA with a single Clausius η.

    t_ij(ε) = −(1 + ε (Φ_i+Φ_j)/2)
    Φ_i = exp(−r²/(2σ²))
    A = ∑_{cut bonds} 1/|t_ij|

PRE-REGISTERED:
  N=12, R=2, 512 balls. Φ at (6,6,6), σ=2.
  ε=0.02 and 0.04. Ground state of H(ε). E<0 occupancy.
  C_vac   |ρ(δS, Tr(K_vac ΔC))| > 0.95
  C0      max|δS|>1e-6 and max|δA|>1e-4
  C1      Pearson(δS(ε), δS(2ε)) > 0.95
  C_area  |ρ(δS, δA)| > 0.80
  C_eta   PRIMARY gravity. On balls with |δA|>1e-4,
          IQR(δS/δA) / |median(δS/δA)| < 0.35
  C_pred  R(δS, δA) < R(δS, P_CHM)   (area beats energy)
  Matter-only identity: hops fixed ⇒ δA ≡ 0.

Not claimed: 8πG, Einstein equation, de Sitter, MODELS.md.

Writes ../data/m9_30_area.json
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
SIGMA = 2.0
EPS1 = 0.02
EPS2 = 0.04


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def phi_field(n: int, src, sigma: float) -> np.ndarray:
    phi = np.zeros((n**3,), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                phi[idx(x, y, z, n)] = np.exp(-0.5 * rr / (sigma * sigma))
    return phi


def hop_H(n: int, phi: np.ndarray | None, eps: float) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        scale = 1.0
                        if phi is not None:
                            scale = 1.0 + eps * 0.5 * (phi[i] + phi[j])
                        ham[i, j] = ham[j, i] = -scale
    return ham


def tmap(n: int, phi: np.ndarray, eps: float) -> dict:
    out = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        scale = 1.0 + eps * 0.5 * (phi[i] + phi[j])
                        out[(min(i, j), max(i, j))] = -scale
    return out


def occupy(ham: np.ndarray):
    ev, vecs = np.linalg.eigh(ham)
    filled = ev < 0.0
    return vecs[:, filled] @ vecs[:, filled].T, int(filled.sum())


def peschel_s(corr: np.ndarray, sl: np.ndarray) -> float:
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr: np.ndarray, sl: np.ndarray) -> np.ndarray:
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def area_geom(n: int, sl: np.ndarray, tm: dict) -> float:
    inside = set(int(i) for i in sl)
    acc = 0.0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                if i not in inside:
                    continue
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if not (0 <= xx < n and 0 <= yy < n and 0 <= zz < n):
                        acc += 1.0
                        continue
                    j = idx(xx, yy, zz, n)
                    if j not in inside:
                        acc += 1.0 / abs(tm[(min(i, j), max(i, j))])
    return acc


def ball_sites(n: int, radius: int, cx: int, cy: int, cz: int) -> np.ndarray:
    r2 = radius * radius
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2:
                    sl.append(idx(x, y, z, n))
    return np.array(sl, dtype=int)


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


def main() -> int:
    phi = phi_field(N, SRC, SIGMA)
    ham0 = hop_H(N, None, 0.0)
    ham1 = hop_H(N, phi, EPS1)
    ham2 = hop_H(N, phi, EPS2)
    c0, n0 = occupy(ham0)
    c1, n1 = occupy(ham1)
    c2, n2 = occupy(ham2)
    de = np.sum(ham1 * c1, axis=1) - np.sum(ham0 * c0, axis=1)
    dc = c1 - c0
    tm0 = tmap(N, phi, 0.0)
    tm1 = tmap(N, phi, EPS1)
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds1, ds2, da, pchm, pflat, pk = [], [], [], [], [], []
    for cx, cy, cz in centers:
        sl = ball_sites(N, RADIUS, cx, cy, cz)
        ds1.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        ds2.append(peschel_s(c2, sl) - peschel_s(c0, sl))
        da.append(area_geom(N, sl, tm1) - area_geom(N, sl, tm0))
        pk.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
        s_c = s_f = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        e = de[idx(x, y, z)]
                        s_c += (r2max - rr) * e
                        s_f += e
        pchm.append(s_c)
        pflat.append(s_f)
    ds1 = np.asarray(ds1, dtype=float)
    ds2 = np.asarray(ds2, dtype=float)
    da = np.asarray(da, dtype=float)
    pchm = np.asarray(pchm, dtype=float)
    pflat = np.asarray(pflat, dtype=float)
    pk = np.asarray(pk, dtype=float)
    mask = np.abs(da) > 1e-4
    eta = ds1[mask] / da[mask]
    med = float(np.median(eta))
    iqr = float(np.percentile(eta, 75) - np.percentile(eta, 25))
    rel_iqr = float(iqr / abs(med)) if med != 0.0 else float("inf")
    r_area = residual_ratio(ds1, da)
    r_chm = residual_ratio(ds1, pchm)
    r_flat = residual_ratio(ds1, pflat)
    rho_k = pearson(ds1, pk)
    c_vac = bool(abs(rho_k) > 0.95)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6 and float(np.max(np.abs(da))) > 1e-4)
    c1g = bool(pearson(ds1, ds2) > 0.95)
    c_area = bool(abs(pearson(ds1, da)) > 0.80)
    c_eta = bool(rel_iqr < 0.35)
    c_pred = bool(r_area < r_chm)
    gravity = bool(c_vac and c0g and c1g and c_area and c_eta)
    if not c_vac:
        verdict = "INSTRUMENT_REJECT"
    elif gravity:
        verdict = "CLAUSIUS_CONSTANT_ETA"
    elif c_area:
        verdict = "AREA_CORRELATES_NOT_CLAUSIUS"
    else:
        verdict = "NO_AREA_LAW"
    payload = {
        "task": "m9.30_area",
        "n_balls": int(len(centers)),
        "n_occ": [int(n0), int(n1), int(n2)],
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "max_abs_dA": float(np.max(np.abs(da))),
        "rho_eps": pearson(ds1, ds2),
        "rho_Kvac": rho_k,
        "rho_area": pearson(ds1, da),
        "rho_CHM": pearson(ds1, pchm),
        "rho_flat": pearson(ds1, pflat),
        "R_area": r_area,
        "R_CHM": r_chm,
        "R_flat": r_flat,
        "eta_median": med,
        "eta_rel_iqr": rel_iqr,
        "n_eta": int(mask.sum()),
        "matter_only_dA": 0.0,
        "C_vac": c_vac,
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C_area": c_area,
        "C_eta_PRIMARY": c_eta,
        "C_pred_area_beats_CHM": c_pred,
        "all_gates": gravity,
        "verdict": verdict,
        "not_claimed": ["8pi G", "Einstein equation", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_30_area.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if gravity else 1


if __name__ == "__main__":
    raise SystemExit(main())
