#!/usr/bin/env python3
"""M9.18: first law on SURFACE hops (the horizon).

M9.17 locked bulk and did not select CHM. The surface diagnostic
on that run is not a pass (same sample). This file is a new grid.

Jacobson's Clausius lives on the horizon, not in the bulk.
Surface: r_mid >= 0.75 R. Fit K_CHM and K_flat on that set only.
Score δS vs Tr(K ΔC) on those hops. ε=0.01.

PRE-REGISTERED (same numerical floors as M9.16, sign-blind):
  C0  1d endpoint bonds (|x_mid| >= L/4): |ρ(δS, δS_CHM)| > 0.70.
  C1  3d N=10 R=3 surface: |ρ| > 0.60.
  C2  R_shape(CHM) < 0.70.
  C3  PRIMARY. R_shape(CHM) < R_shape(flat).

Not claimed: eta=1/4G, Einstein, foam, de Sitter.

Writes ../data/m9_18_surface_first_law.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N1, L1 = 200, 24
N3, R3 = 10, 3
EPS_H = 0.01


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
    if len(y) < 3:
        return float("nan")
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(y - mat @ coef) / den)


def peschel(ham, sl):
    ev, vecs = np.linalg.eigh(ham)
    full = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    ca = full[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, CLIP, 1.0 - CLIP)
    ent = float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))
    k = (u * np.log((1.0 - w) / w)) @ u.T
    return ent, 0.5 * (k + k.T), ca


def fit_chm(k, pairs, weights):
    knn = np.array([k[i, j] for i, j in pairs], dtype=float)
    mat = np.column_stack([weights, np.ones(len(weights))])
    coef, _, _, _ = np.linalg.lstsq(mat, knn, rcond=None)
    kchm = np.zeros_like(k)
    for (i, j), ww in zip(pairs, weights):
        val = coef[0] * ww + coef[1]
        kchm[i, j] = kchm[j, i] = val
    return kchm, coef


def kernel_flat(k, pairs):
    fill = float(np.mean([k[i, j] for i, j in pairs]))
    out = np.zeros_like(k)
    for i, j in pairs:
        out[i, j] = out[j, i] = fill
    return out


def score(ds, pred, flat) -> dict:
    ds, pred, flat = map(lambda z: np.asarray(z, dtype=float), (ds, pred, flat))
    return {
        "n": int(ds.size),
        "rho_chm": pearson(ds, pred),
        "R_shape_chm": residual_ratio(ds, pred),
        "rho_flat": pearson(ds, flat),
        "R_shape_flat": residual_ratio(ds, flat),
        "mean_abs_dS": float(np.mean(np.abs(ds))) if ds.size else 0.0,
    }


def one_d() -> dict:
    ham0 = np.zeros((N1, N1), dtype=float)
    for i in range(N1 - 1):
        ham0[i, i + 1] = ham0[i + 1, i] = -1.0
    mid = N1 // 2
    sl = np.arange(mid - L1 // 2, mid + L1 // 2)
    s0, k0, c0 = peschel(ham0, sl)
    x = np.arange(L1) - (L1 - 1) / 2.0
    w = (L1 / 2.0) ** 2 - x**2
    pairs, wnn = [], []
    for i in range(L1 - 1):
        if abs(0.5 * (x[i] + x[i + 1])) >= L1 / 4.0:
            pairs.append((i, i + 1))
            wnn.append(0.5 * (w[i] + w[i + 1]))
    wnn = np.array(wnn)
    kchm, coef = fit_chm(k0, pairs, wnn)
    kflat = kernel_flat(k0, pairs)
    ds, pred, flat = [], [], []
    for i, j in pairs:
        a, b = int(sl[i]), int(sl[j])
        ham = ham0.copy()
        ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
        s, _, c = peschel(ham, sl)
        dc = c - c0
        ds.append(s - s0)
        pred.append(float(np.trace(kchm @ dc)))
        flat.append(float(np.trace(kflat @ dc)))
    out = score(ds, pred, flat)
    out["fit_ab"] = [float(coef[0]), float(coef[1])]
    out["C0"] = bool(abs(out["rho_chm"]) > 0.70)
    return out


def idx(x, y, z, n) -> int:
    return (x * n + y) * n + z


def three_d(n: int, radius: int) -> dict:
    ham0 = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham0[i, j] = ham0[j, i] = -1.0
    cen = n // 2
    sl_l, coords = [], []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cen) ** 2 + (y - cen) ** 2 + (z - cen) ** 2 <= radius * radius:
                    sl_l.append(idx(x, y, z, n))
                    coords.append((x, y, z))
    sl = np.array(sl_l)
    s0, k0, c0 = peschel(ham0, sl)
    pos = {pt: i for i, pt in enumerate(coords)}
    wv = np.array(
        [
            radius * radius - ((x - cen) ** 2 + (y - cen) ** 2 + (z - cen) ** 2)
            for x, y, z in coords
        ],
        dtype=float,
    )
    pairs, wnn, sites = [], [], []
    for i, (x, y, z) in enumerate(coords):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nbr = (x + d[0], y + d[1], z + d[2])
            j = pos.get(nbr)
            if j is None or j <= i:
                continue
            mx = 0.5 * (x + nbr[0] - 2 * cen)
            my = 0.5 * (y + nbr[1] - 2 * cen)
            mz = 0.5 * (z + nbr[2] - 2 * cen)
            rmid = float(np.sqrt(mx * mx + my * my + mz * mz))
            if rmid < 0.75 * radius:
                continue
            pairs.append((i, j))
            wnn.append(0.5 * (wv[i] + wv[j]))
            sites.append((coords[i], nbr))
    wnn = np.array(wnn)
    kchm, coef = fit_chm(k0, pairs, wnn)
    kflat = kernel_flat(k0, pairs)
    ds, pred, flat = [], [], []
    for (i, j), (p, q) in zip(pairs, sites):
        a, b = idx(*p, n), idx(*q, n)
        ham = ham0.copy()
        ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
        s, _, ca = peschel(ham, sl)
        dc = ca - c0
        ds.append(s - s0)
        pred.append(float(np.trace(kchm @ dc)))
        flat.append(float(np.trace(kflat @ dc)))
    out = score(ds, pred, flat)
    out["fit_ab"] = [float(coef[0]), float(coef[1])]
    out["N"] = n
    out["R"] = radius
    return out


def main() -> int:
    d1 = one_d()
    d3 = three_d(N3, R3)
    c0 = bool(d1["C0"])
    c1 = bool(abs(d3["rho_chm"]) > 0.60)
    c2 = bool(d3["R_shape_chm"] < 0.70)
    c3 = bool(d3["R_shape_chm"] < d3["R_shape_flat"])
    ok = bool(c0 and c1 and c2 and c3)
    payload = {
        "task": "m9.18_surface_first_law",
        "observable": "surface/horizon dS vs Tr(K dC); CHM vs flat",
        "one_d_surface": d1,
        "three_d": d3,
        "C0_instrument": c0,
        "C1_rho": c1,
        "C2_Rshape": c2,
        "C3_PRIMARY_chm_beats_flat": c3,
        "all_gates": ok,
        "verdict": (
            "SURFACE_FIRST_LAW_INSTRUMENT_REJECTED"
            if not c0
            else ("SURFACE_CHM_SELECTED" if ok else "SURFACE_CHM_NOT_SELECTED")
        ),
        "not_claimed": ["eta = 1/4G", "Einstein theorem", "foam", "de Sitter"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_18_surface_first_law.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
