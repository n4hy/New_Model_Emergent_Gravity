#!/usr/bin/env python3
"""M9.22: score C4 at R=5 with FIXED half-filling.

M9.21 ev<0 filling flipped n_occ on 25-27 of 30 hops. C4
unscored. This instrument occupies the lowest V/2 eigenstates
always, so occupancy cannot flip. Same grid, bins, kernels, C4.

PRE-REGISTERED:
  N=14 R=5, 10 lex bonds x 3 r_mid bins, ε=0.01.
  Occupy lowest ham.shape[0]//2 states (not ev<0).
  C3  R_CHM < R_flat.
  C4  PRIMARY. R_CHM < R_lin. A tie |R_CHM-R_lin|<0.005 is
      scored TIE, not rejected.

Not claimed: Planck, eta=1/4G, Einstein, Bloch, de Sitter.

Writes ../data/m9_22_halffill_horizon.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N1, L1 = 200, 24
N3, R3 = 14, 5
EPS_H = 0.01
BINS = ((3.75, 4.20), (4.20, 4.60), (4.60, 5.01))
PER_BIN = 10


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
    nocc = ham.shape[0] // 2
    full = vecs[:, :nocc] @ vecs[:, :nocc].T
    ca = full[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, CLIP, 1.0 - CLIP)
    ent = float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))
    k = (u * np.log((1.0 - w) / w)) @ u.T
    return ent, 0.5 * (k + k.T), ca, nocc


def fit_weight(k, pairs, weights):
    knn = np.array([k[i, j] for i, j in pairs], dtype=float)
    mat = np.column_stack([weights, np.ones(len(weights))])
    coef, _, _, _ = np.linalg.lstsq(mat, knn, rcond=None)
    out = np.zeros_like(k)
    for (i, j), ww in zip(pairs, weights):
        out[i, j] = out[j, i] = coef[0] * ww + coef[1]
    return out


def kernel_flat(k, pairs):
    fill = float(np.mean([k[i, j] for i, j in pairs]))
    out = np.zeros_like(k)
    for i, j in pairs:
        out[i, j] = out[j, i] = fill
    return out


def score(ds, pc, pl, pf) -> dict:
    return {
        "n": int(len(ds)),
        "rho_chm": pearson(ds, pc),
        "R_chm": residual_ratio(ds, pc),
        "rho_lin": pearson(ds, pl),
        "R_lin": residual_ratio(ds, pl),
        "rho_flat": pearson(ds, pf),
        "R_flat": residual_ratio(ds, pf),
    }


def one_d() -> dict:
    ham0 = np.zeros((N1, N1), dtype=float)
    for i in range(N1 - 1):
        ham0[i, i + 1] = ham0[i + 1, i] = -1.0
    mid = N1 // 2
    sl = np.arange(mid - L1 // 2, mid + L1 // 2)
    s0, k0, c0, n0 = peschel(ham0, sl)
    x = np.arange(L1) - (L1 - 1) / 2.0
    kept, ds, dcs, wc, wl = [], [], [], [], []
    for i in range(L1 - 1):
        xm = 0.5 * (x[i] + x[i + 1])
        if abs(xm) < L1 / 4.0:
            continue
        a, b = int(sl[i]), int(sl[i + 1])
        ham = ham0.copy()
        ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
        s, _, c, n1 = peschel(ham, sl)
        if n1 != n0:
            continue
        kept.append((i, i + 1))
        ds.append(s - s0)
        dcs.append(c - c0)
        wc.append((L1 / 2.0) ** 2 - xm**2)
        wl.append(L1 / 2.0 - abs(xm))
    kchm = fit_weight(k0, kept, np.array(wc))
    klin = fit_weight(k0, kept, np.array(wl))
    kfl = kernel_flat(k0, kept)
    out = score(
        ds,
        [float(np.trace(kchm @ d)) for d in dcs],
        [float(np.trace(klin @ d)) for d in dcs],
        [float(np.trace(kfl @ d)) for d in dcs],
    )
    out["C0"] = bool(abs(out["rho_chm"]) > 0.70)
    out["n_occ0"] = n0
    return out


def idx(x, y, z, n) -> int:
    return (x * n + y) * n + z


def sample_surface(n: int, radius: int, per_bin: int):
    cen = n // 2
    coords = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cen) ** 2 + (y - cen) ** 2 + (z - cen) ** 2 <= radius * radius:
                    coords.append((x, y, z))
    pos = {p: i for i, p in enumerate(coords)}
    buckets = [[] for _ in BINS]
    for x, y, z in coords:
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nbr = (x + d[0], y + d[1], z + d[2])
            j = pos.get(nbr)
            i = pos[(x, y, z)]
            if j is None or j <= i:
                continue
            mx = 0.5 * (x + nbr[0] - 2 * cen)
            my = 0.5 * (y + nbr[1] - 2 * cen)
            mz = 0.5 * (z + nbr[2] - 2 * cen)
            rm = float(np.sqrt(mx * mx + my * my + mz * mz))
            for b, (lo, hi) in enumerate(BINS):
                if lo <= rm < hi:
                    buckets[b].append((x, y, z, d[0], d[1], d[2], i, j, rm, nbr))
                    break
    picked = []
    for b, bucket in enumerate(buckets):
        bucket.sort()
        picked.extend(bucket[:per_bin])
    return coords, picked


def three_d(n: int, radius: int, per_bin: int) -> dict:
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
    coords, picked = sample_surface(n, radius, per_bin)
    sl = np.array([idx(*p, n) for p in coords])
    s0, k0, c0, n0 = peschel(ham0, sl)
    kept, ds, dcs, rmid = [], [], [], []
    n_flip = 0
    for x, y, z, dx, dy, dz, i, j, rm, nbr in picked:
        a, b = idx(x, y, z, n), idx(*nbr, n)
        ham = ham0.copy()
        ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
        s, _, c, n1 = peschel(ham, sl)
        if n1 != n0:
            n_flip += 1
            continue
        kept.append((i, j))
        ds.append(s - s0)
        dcs.append(c - c0)
        rmid.append(rm)
    rmid = np.array(rmid, dtype=float)
    kchm = fit_weight(k0, kept, radius * radius - rmid**2)
    klin = fit_weight(k0, kept, radius - rmid)
    kfl = kernel_flat(k0, kept)
    out = score(
        ds,
        [float(np.trace(kchm @ d)) for d in dcs],
        [float(np.trace(klin @ d)) for d in dcs],
        [float(np.trace(kfl @ d)) for d in dcs],
    )
    out.update(
        {
            "N": n,
            "R": radius,
            "n_tried": int(len(picked)),
            "n_flip": int(n_flip),
            "n_occ0": n0,
            "per_bin": per_bin,
        }
    )
    return out


def main() -> int:
    d1 = one_d()
    d3 = three_d(N3, R3, PER_BIN)
    c0 = bool(d1["C0"])
    c1 = bool(abs(d3["rho_chm"]) > 0.60)
    c2 = bool(d3["R_chm"] < 0.70)
    c3 = bool(d3["R_chm"] < d3["R_flat"])
    gap = abs(d3["R_chm"] - d3["R_lin"])
    c4_tie = bool(gap < 0.005)
    c4_pass = bool(d3["R_chm"] < d3["R_lin"] and not c4_tie)
    c4_scored = bool(d3["n"] >= 10)
    if not c4_scored:
        c4_label = "NOT_SCORED"
    elif c4_tie:
        c4_label = "TIE"
    elif c4_pass:
        c4_label = "PASS"
    else:
        c4_label = "FAIL"
    payload = {
        "task": "m9.22_halffill_horizon",
        "observable": "R=5 half-fill; CHM vs linear vs flat; C4 scored",
        "one_d": d1,
        "three_d": d3,
        "C0_instrument": c0,
        "C1_rho": c1,
        "C2_Rshape": c2,
        "C3_beats_flat": c3,
        "C4_PRIMARY": c4_label,
        "C4_gap": gap,
        "C4_scored": c4_scored,
        "all_gates": bool(c0 and c1 and c2 and c3 and c4_pass),
        "verdict": f"C4_{c4_label}",
        "not_claimed": ["Planck", "eta=1/4G", "Einstein", "Bloch", "de Sitter"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_22_halffill_horizon.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c4_scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
