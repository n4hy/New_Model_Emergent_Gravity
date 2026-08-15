#!/usr/bin/env python3
"""Surface first-law audit. No solver import.

1d N=160 L=20 endpoints. 3d N=12 R=3, r_mid >= 2.25. ε=0.015.

Writes ../data/m9_18_audit_surface.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
EPS_H = 0.015


def pearson(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if den == 0.0 else float(np.dot(a, b) / den)


def rshape(y, x) -> float:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if len(y) < 3:
        return float("nan")
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    return float("nan") if den == 0.0 else float(np.linalg.norm(y - mat @ coef) / den)


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
    knn = np.array([k[i, j] for i, j in pairs])
    mat = np.column_stack([weights, np.ones(len(weights))])
    coef, _, _, _ = np.linalg.lstsq(mat, knn, rcond=None)
    kchm = np.zeros_like(k)
    for (i, j), ww in zip(pairs, weights):
        val = coef[0] * ww + coef[1]
        kchm[i, j] = kchm[j, i] = val
    return kchm


def kernel_flat(k, pairs):
    fill = float(np.mean([k[i, j] for i, j in pairs]))
    out = np.zeros_like(k)
    for i, j in pairs:
        out[i, j] = out[j, i] = fill
    return out


def zone_run(n, radius, rmin_frac, ell=None):
    raise RuntimeError("unused")


def audit_1d() -> dict:
    n, ell = 160, 20
    ham0 = np.zeros((n, n))
    for i in range(n - 1):
        ham0[i, i + 1] = ham0[i + 1, i] = -1.0
    mid = n // 2
    sl = np.arange(mid - ell // 2, mid + ell // 2)
    s0, k0, c0 = peschel(ham0, sl)
    x = np.arange(ell) - (ell - 1) / 2.0
    w = (ell / 2.0) ** 2 - x**2
    pairs, wnn = [], []
    for i in range(ell - 1):
        if abs(0.5 * (x[i] + x[i + 1])) >= ell / 4.0:
            pairs.append((i, i + 1))
            wnn.append(0.5 * (w[i] + w[i + 1]))
    wnn = np.array(wnn)
    kchm = fit_chm(k0, pairs, wnn)
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
    return {
        "n": int(len(ds)),
        "rho_chm": pearson(ds, pred),
        "R_shape_chm": rshape(ds, pred),
        "rho_flat": pearson(ds, flat),
        "R_shape_flat": rshape(ds, flat),
        "C0": bool(abs(pearson(ds, pred)) > 0.70),
    }


def audit_3d() -> dict:
    n, radius = 12, 3

    def ix(x, y, z):
        return (x * n + y) * n + z

    ham0 = np.zeros((n**3, n**3))
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = ix(x, y, z)
                if x + 1 < n:
                    ham0[i, ix(x + 1, y, z)] = ham0[ix(x + 1, y, z), i] = -1.0
                if y + 1 < n:
                    ham0[i, ix(x, y + 1, z)] = ham0[ix(x, y + 1, z), i] = -1.0
                if z + 1 < n:
                    ham0[i, ix(x, y, z + 1)] = ham0[ix(x, y, z + 1), i] = -1.0
    cen = n // 2
    sl_l, coords = [], []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cen) ** 2 + (y - cen) ** 2 + (z - cen) ** 2 <= radius * radius:
                    sl_l.append(ix(x, y, z))
                    coords.append((x, y, z))
    sl = np.array(sl_l)
    s0, k0, c0 = peschel(ham0, sl)
    pos = {pt: i for i, pt in enumerate(coords)}
    wv = np.array(
        [radius * radius - ((x - cen) ** 2 + (y - cen) ** 2 + (z - cen) ** 2) for x, y, z in coords]
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
    kchm = fit_chm(k0, pairs, wnn)
    kflat = kernel_flat(k0, pairs)
    ds, pred, flat = [], [], []
    for (i, j), (p, q) in zip(pairs, sites):
        a, b = ix(*p), ix(*q)
        ham = ham0.copy()
        ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
        s, _, ca = peschel(ham, sl)
        dc = ca - c0
        ds.append(s - s0)
        pred.append(float(np.trace(kchm @ dc)))
        flat.append(float(np.trace(kflat @ dc)))
    rs_c, rs_f = rshape(ds, pred), rshape(ds, flat)
    rho = pearson(ds, pred)
    return {
        "n": int(len(ds)),
        "rho_chm": rho,
        "R_shape_chm": rs_c,
        "rho_flat": pearson(ds, flat),
        "R_shape_flat": rs_f,
        "C1": bool(abs(rho) > 0.60),
        "C2": bool(rs_c < 0.70),
        "C3": bool(rs_c < rs_f),
    }


def main() -> int:
    d1 = audit_1d()
    d3 = audit_3d()
    ok = bool(d1["C0"] and d3["C1"] and d3["C2"] and d3["C3"])
    payload = {
        "task": "m9.18_audit_surface",
        "method": "surface; 1d N=160 L=20 ends; 3d N=12 R=3 r_mid>=2.25",
        "one_d": d1,
        "three_d": d3,
        "verdicts": {"C3_surface": "CONFIRMED" if ok else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_18_audit_surface.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
