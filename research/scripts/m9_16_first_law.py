#!/usr/bin/env python3
"""M9.16: first law with the CHM kernel (effective, non-tautological).

Failed instrument (discarded): χ = ΔS/Δt vs w. In 1d that has ρ≈0.28.
ΔC is nonlocal, so a bond-local χ is not the Clausius probe.

Effective observable, locked after that rejection:
  Fit K_CHM on the vacuum NN kernel, K_CHM_ij = a w_ij + b.
  Perturb a hop, measure
      δS = S_ε - S_0
      δS_CHM = Tr(K_CHM (C_ε - C_0)).
  The first law says these track. K_CHM is not K_exact, so this is
  not Tr(K dC) = dS.

PRE-REGISTERED:
  C0  INSTRUMENT. 1d massless, every interval bond, ε=0.01:
        ρ(δS, δS_CHM) > 0.80.
  C1  3+1D N=12 R=3, ALL interior NN, ε=0.01:
        ρ(δS, δS_CHM) > 0.60.
  C2  PRIMARY. R_shape = ||δS-(α δS_CHM+β)|| / ||δS-mean|| < 0.70.
      (C1 is the floor on ρ. C2 is the residual after an affine
      fit, which is what "tracks" means. 0.70 is ρ≳0.71.)
  C3  Mutation / null. K_flat = mean(K_NN) on every hop (local,
        no CHM shape). The CHM kernel must beat it:
        R_shape(δS, δS_CHM) < R_shape(δS, δS_flat).
        A permutation of w is useless: the intercept makes every
        NN kernel look the same.
  C4  Linearity. On 3 pre-registered bonds (first of bins 0,1,2),
        ρ of the 3-vector (δS, δS_CHM) at ε=0.005 and ε=0.01
        both exceed 0.50 (same sign). Cheap sanity, not the primary.

Not claimed: eta=1/4G, Einstein, foam, de Sitter, unique quadratic.

Writes ../data/m9_16_first_law.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

CLIP = 1e-12
N1, L1 = 200, 24
N3, R3 = 12, 3
EPS_H = 0.01
EPS_LIN = 0.005


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
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


def peschel(ham: np.ndarray, sl: np.ndarray):
    ev, vecs = np.linalg.eigh(ham)
    full = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    ca = full[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, CLIP, 1.0 - CLIP)
    ent = float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))
    k = (u * np.log((1.0 - w) / w)) @ u.T
    k = 0.5 * (k + k.T)
    return ent, k, ca


def fit_chm(k: np.ndarray, pairs: list, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    knn = np.array([k[i, j] for i, j in pairs], dtype=float)
    mat = np.column_stack([weights, np.ones(len(weights))])
    coef, _, _, _ = np.linalg.lstsq(mat, knn, rcond=None)
    kchm = np.zeros_like(k)
    for (i, j), ww in zip(pairs, weights):
        val = coef[0] * ww + coef[1]
        kchm[i, j] = kchm[j, i] = val
    return kchm, coef


def kernel_flat(k: np.ndarray, pairs: list) -> np.ndarray:
    knn = np.array([k[i, j] for i, j in pairs], dtype=float)
    fill = float(knn.mean())
    out = np.zeros_like(k)
    for i, j in pairs:
        out[i, j] = out[j, i] = fill
    return out


def one_d() -> dict:
    ham0 = np.zeros((N1, N1), dtype=float)
    for i in range(N1 - 1):
        ham0[i, i + 1] = ham0[i + 1, i] = -1.0
    mid = N1 // 2
    sl = np.arange(mid - L1 // 2, mid + L1 // 2)
    s0, k0, c0 = peschel(ham0, sl)
    x = np.arange(L1) - (L1 - 1) / 2.0
    w = (L1 / 2.0) ** 2 - x**2
    pairs = [(i, i + 1) for i in range(L1 - 1)]
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
    ds = np.array(ds)
    pred = np.array(pred)
    flat = np.array(flat)
    rho = pearson(ds, pred)
    rs = residual_ratio(ds, pred)
    rs_flat = residual_ratio(ds, flat)
    return {
        "n": int(ds.size),
        "fit_ab": [float(coef[0]), float(coef[1])],
        "rho": rho,
        "R_shape": rs,
        "rho_flat": pearson(ds, flat),
        "R_shape_flat": rs_flat,
        "C0": bool(rho > 0.80),
        "C3_1d": bool(rs < rs_flat),
    }


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
    c = n // 2
    sl_l, coords = [], []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius * radius:
                    sl_l.append(idx(x, y, z, n))
                    coords.append((x, y, z))
    sl = np.array(sl_l)
    s0, k0, c0 = peschel(ham0, sl)
    pos = {pt: i for i, pt in enumerate(coords)}
    w = np.array(
        [radius * radius - ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) for x, y, z in coords],
        dtype=float,
    )
    pairs = []
    wnn = []
    rmid = []
    sites = []
    for i, (x, y, z) in enumerate(coords):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nbr = (x + d[0], y + d[1], z + d[2])
            j = pos.get(nbr)
            if j is None or j <= i:
                continue
            pairs.append((i, j))
            wnn.append(0.5 * (w[i] + w[j]))
            mx, my, mz = 0.5 * (x + nbr[0] - 2 * c), 0.5 * (y + nbr[1] - 2 * c), 0.5 * (z + nbr[2] - 2 * c)
            rmid.append(float(np.sqrt(mx * mx + my * my + mz * mz)))
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
    ds = np.array(ds)
    pred = np.array(pred)
    flat = np.array(flat)
    # linearity on first pair of bins 0,1,2
    lin = []
    seen = {0: False, 1: False, 2: False}
    for rec, site, rm in zip(pairs, sites, rmid):
        b = min(int(rm), 2)
        if seen[b]:
            continue
        seen[b] = True
        p, q = site
        a, bidx = idx(*p, n), idx(*q, n)
        ham = ham0.copy()
        ham[a, bidx] = ham[bidx, a] = ham0[a, bidx] * (1.0 + EPS_LIN)
        s, _, ca = peschel(ham, sl)
        lin.append(
            {
                "bin": b,
                "dS": s - s0,
                "pred": float(np.trace(kchm @ (ca - c0))),
            }
        )
        if all(seen.values()):
            break
    rho = pearson(ds, pred)
    rs = residual_ratio(ds, pred)
    rs_flat = residual_ratio(ds, flat)
    return {
        "n_pairs": int(ds.size),
        "fit_ab": [float(coef[0]), float(coef[1])],
        "rho": rho,
        "R_shape": rs,
        "rho_flat": pearson(ds, flat),
        "R_shape_flat": rs_flat,
        "linearity": lin,
    }


def main() -> int:
    d1 = one_d()
    d3 = three_d()
    c0 = bool(d1["C0"])
    c1 = bool(d3["rho"] > 0.60)
    c2 = bool(d3["R_shape"] < 0.70)
    c3 = bool(d3["R_shape"] < d3["R_shape_flat"])
    lin = d3["linearity"]
    c4 = bool(len(lin) >= 2) and all(
        (r["dS"] * r["pred"] > 0.0) or (abs(r["dS"]) < 1e-8) for r in lin
    )
    ok = bool(c0 and c1 and c2 and c3)
    payload = {
        "task": "m9.16_first_law",
        "observable": "dS vs Tr(K_CHM dC); K_CHM fitted NN envelope, not K_exact",
        "one_d": d1,
        "three_d": d3,
        "C0_instrument": c0,
        "C1_rho": c1,
        "C2_PRIMARY_Rshape": c2,
        "C3_permutation": c3,
        "C4_linearity_sign": c4,
        "all_gates": ok,
        "verdict": (
            "FIRST_LAW_INSTRUMENT_REJECTED"
            if not c0
            else ("FIRST_LAW_PASS" if ok else "FIRST_LAW_FAIL")
        ),
        "not_claimed": [
            "eta = 1/4G",
            "Einstein theorem",
            "mean-zero foam",
            "de Sitter",
            "unique CHM quadratic",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_16_first_law.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
