#!/usr/bin/env python3
"""M9.17: first law on BULK hops only (surface drowned Paper 26).

Same non-tautological observable as M9.16:
    δS vs Tr(K (C_ε - C_0)), K_CHM vs K_flat.
Paper 26 scored every hop. |δS| is largest at the cut, so a
shapeless local K can win by matching the surface. This run
restricts the score to bulk bonds.

PRE-REGISTERED:
  Bulk: midpoint radius r_mid <= 0.5 R  (3d), |x_mid| <= L/4 (1d).
  Surface control: r_mid >= 0.75 R. Not a pass gate.
  K_CHM and K_flat are FIT on the same bulk NN set they are
  scored on (fair: surface intercept cannot help).
  ε = 0.01. Complete census of bulk bonds, no cherry-pick.

  C0  1d central bonds: ρ(δS, δS_CHM) > 0.70.
  C1  3d bulk: ρ(δS, δS_CHM) > 0.60. N=12, R=4.
  C2  3d bulk R_shape(CHM) < 0.70.
  C3  PRIMARY. Bulk: R_shape(CHM) < R_shape(flat).
  C4  Diagnostic only. Surface: report who wins. Not a gate.

Not claimed: eta=1/4G, Einstein, foam, de Sitter.

Writes ../data/m9_17_bulk_first_law.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

CLIP = 1e-12
N1, L1 = 200, 24
N3, R3 = 12, 4
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


def peschel(ham: np.ndarray, sl: np.ndarray):
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
    ds = np.asarray(ds, dtype=float)
    pred = np.asarray(pred, dtype=float)
    flat = np.asarray(flat, dtype=float)
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
    pairs_all = [(i, i + 1) for i in range(L1 - 1)]
    xmid = np.array([0.5 * (x[i] + x[j]) for i, j in pairs_all])
    bulk_idx = [n for n, xm in enumerate(xmid) if abs(xm) <= L1 / 4.0]
    pairs = [pairs_all[n] for n in bulk_idx]
    wnn = np.array([0.5 * (w[i] + w[j]) for i, j in pairs])
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
    out["C0"] = bool(out["rho_chm"] > 0.70)
    return out


def idx(x, y, z, n) -> int:
    return (x * n + y) * n + z


def three_d() -> dict:
    n, radius = N3, R3
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
    records = []
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
            records.append(
                {
                    "pair": (i, j),
                    "sites": (coords[i], nbr),
                    "w": 0.5 * (wv[i] + wv[j]),
                    "rmid": rmid,
                }
            )
    bulk = [r for r in records if r["rmid"] <= 0.5 * radius]
    surf = [r for r in records if r["rmid"] >= 0.75 * radius]

    def run_zone(zone):
        pairs = [r["pair"] for r in zone]
        wnn = np.array([r["w"] for r in zone], dtype=float)
        kchm, coef = fit_chm(k0, pairs, wnn)
        kflat = kernel_flat(k0, pairs)
        ds, pred, flat, rmids = [], [], [], []
        for r in zone:
            p, q = r["sites"]
            a, b = idx(*p, n), idx(*q, n)
            ham = ham0.copy()
            ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
            s, _, ca = peschel(ham, sl)
            dc = ca - c0
            ds.append(s - s0)
            pred.append(float(np.trace(kchm @ dc)))
            flat.append(float(np.trace(kflat @ dc)))
            rmids.append(r["rmid"])
        out = score(ds, pred, flat)
        out["fit_ab"] = [float(coef[0]), float(coef[1])]
        out["rmid_min"] = float(min(rmids)) if rmids else None
        out["rmid_max"] = float(max(rmids)) if rmids else None
        return out

    bulk_out = run_zone(bulk)
    surf_out = run_zone(surf) if surf else {"n": 0}
    return {
        "n_all_nn": int(len(records)),
        "bulk": bulk_out,
        "surface": surf_out,
    }


def main() -> int:
    d1 = one_d()
    d3 = three_d()
    b = d3["bulk"]
    c0 = bool(d1["C0"])
    c1 = bool(b["rho_chm"] > 0.60)
    c2 = bool(b["R_shape_chm"] < 0.70)
    c3 = bool(b["R_shape_chm"] < b["R_shape_flat"])
    ok = bool(c0 and c1 and c2 and c3)
    payload = {
        "task": "m9.17_bulk_first_law",
        "observable": "bulk-only dS vs Tr(K dC); CHM vs flat, both fit on bulk",
        "one_d_bulk": d1,
        "three_d": d3,
        "C0_instrument": c0,
        "C1_rho": c1,
        "C2_Rshape": c2,
        "C3_PRIMARY_chm_beats_flat": c3,
        "all_gates": ok,
        "verdict": (
            "BULK_FIRST_LAW_INSTRUMENT_REJECTED"
            if not c0
            else ("BULK_CHM_SELECTED" if ok else "BULK_CHM_NOT_SELECTED")
        ),
        "not_claimed": [
            "eta = 1/4G",
            "Einstein theorem",
            "mean-zero foam",
            "de Sitter",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_17_bulk_first_law.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
