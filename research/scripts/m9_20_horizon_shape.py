#!/usr/bin/env python3
"""M9.20: horizon first law, CHM vs flat vs linear, stable occupancy.

Paper 27: on the cut, CHM beat flat; tracking was weak; the
parabola was not tested against a linear radial weight.
Paper 25: on K itself, linear can beat CHM on small balls.

This run asks the Clausius question that can still fail:
does the CHM *parabola* beat both a flat kernel and a linear
R-r weight, using only hops that do not flip n_occ?

PRE-REGISTERED:
  Surface: r_mid >= 0.75 R. New grid N=12, R=4 (not Paper 27's R=3).
  Discard any hop with n_occ(ε) != n_occ(0).
  Three kernels fit on the kept set only:
    K_CHM:    a (R^2 - r_mid^2) + b
    K_lin:    a (R - r_mid) + b
    K_flat:   mean(K_NN)
  Score δS vs Tr(K ΔC). Sign-blind Pearson.

  C0  1d ends |x_mid|>=L/4, |ρ_CHM| > 0.70, n_occ stable.
  C1  3d |ρ_CHM| > 0.60.
  C2  R_shape(CHM) < 0.70.
  C3  R_shape(CHM) < R_shape(flat).   (Paper 27 replicate)
  C4  PRIMARY. R_shape(CHM) < R_shape(linear).

Not claimed: eta=1/4G, Einstein, foam, Bloch, de Sitter.

Writes ../data/m9_20_horizon_shape.json
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


def peschel(ham, sl):
    ev, vecs = np.linalg.eigh(ham)
    nocc = int(np.sum(ev < 0.0))
    full = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
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
        val = coef[0] * ww + coef[1]
        out[i, j] = out[j, i] = val
    return out, coef


def kernel_flat(k, pairs):
    fill = float(np.mean([k[i, j] for i, j in pairs]))
    out = np.zeros_like(k)
    for i, j in pairs:
        out[i, j] = out[j, i] = fill
    return out


def score(ds, pred_c, pred_l, pred_f) -> dict:
    ds = np.asarray(ds, dtype=float)
    return {
        "n": int(ds.size),
        "rho_chm": pearson(ds, pred_c),
        "R_chm": residual_ratio(ds, pred_c),
        "rho_lin": pearson(ds, pred_l),
        "R_lin": residual_ratio(ds, pred_l),
        "rho_flat": pearson(ds, pred_f),
        "R_flat": residual_ratio(ds, pred_f),
    }


def one_d() -> dict:
    ham0 = np.zeros((N1, N1), dtype=float)
    for i in range(N1 - 1):
        ham0[i, i + 1] = ham0[i + 1, i] = -1.0
    mid = N1 // 2
    sl = np.arange(mid - L1 // 2, mid + L1 // 2)
    s0, k0, c0, n0 = peschel(ham0, sl)
    x = np.arange(L1) - (L1 - 1) / 2.0
    pairs, wchm, wlin, sites = [], [], [], []
    for i in range(L1 - 1):
        xm = 0.5 * (x[i] + x[i + 1])
        if abs(xm) < L1 / 4.0:
            continue
        pairs.append((i, i + 1))
        wchm.append((L1 / 2.0) ** 2 - xm**2)
        wlin.append(L1 / 2.0 - abs(xm))
        sites.append((int(sl[i]), int(sl[i + 1])))
    kept, ds, dc_list, wchm_k, wlin_k = [], [], [], [], []
    for n, ((i, j), (a, b)) in enumerate(zip(pairs, sites)):
        ham = ham0.copy()
        ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
        s, _, c, n1 = peschel(ham, sl)
        if n1 != n0:
            continue
        kept.append((i, j))
        ds.append(s - s0)
        dc_list.append(c - c0)
        wchm_k.append(wchm[n])
        wlin_k.append(wlin[n])
    wchm_k = np.array(wchm_k, dtype=float)
    wlin_k = np.array(wlin_k, dtype=float)
    kchm, _ = fit_weight(k0, kept, wchm_k)
    klin, _ = fit_weight(k0, kept, wlin_k)
    kflat = kernel_flat(k0, kept)
    pc = [float(np.trace(kchm @ dc)) for dc in dc_list]
    pl = [float(np.trace(klin @ dc)) for dc in dc_list]
    pf = [float(np.trace(kflat @ dc)) for dc in dc_list]
    out = score(ds, pc, pl, pf)
    out["n_tried"] = int(len(pairs))
    out["n_occ0"] = n0
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
    s0, k0, c0, n0 = peschel(ham0, sl)
    pos = {pt: i for i, pt in enumerate(coords)}
    recs = []
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
            recs.append((i, j, coords[i], nbr, rmid))
    ds, dc_list, kept, rkept = [], [], [], []
    n_flip = 0
    for i, j, p, q, rmid in recs:
        a, b = idx(*p, n), idx(*q, n)
        ham = ham0.copy()
        ham[a, b] = ham[b, a] = ham0[a, b] * (1.0 + EPS_H)
        s, _, c, n1 = peschel(ham, sl)
        if n1 != n0:
            n_flip += 1
            continue
        kept.append((i, j))
        rkept.append(rmid)
        ds.append(s - s0)
        dc_list.append(c - c0)
    rkept = np.array(rkept, dtype=float)
    wchm = radius * radius - rkept**2
    wlin = radius - rkept
    kchm, _ = fit_weight(k0, kept, wchm)
    klin, _ = fit_weight(k0, kept, wlin)
    kflat = kernel_flat(k0, kept)
    pc = [float(np.trace(kchm @ dc)) for dc in dc_list]
    pl = [float(np.trace(klin @ dc)) for dc in dc_list]
    pf = [float(np.trace(kflat @ dc)) for dc in dc_list]
    out = score(ds, pc, pl, pf)
    out.update(
        {
            "N": n,
            "R": radius,
            "n_tried": int(len(recs)),
            "n_flip": int(n_flip),
            "n_occ0": n0,
        }
    )
    return out


def main() -> int:
    d1 = one_d()
    d3 = three_d(N3, R3)
    c0 = bool(d1["C0"])
    c1 = bool(abs(d3["rho_chm"]) > 0.60)
    c2 = bool(d3["R_chm"] < 0.70)
    c3 = bool(d3["R_chm"] < d3["R_flat"])
    c4 = bool(d3["R_chm"] < d3["R_lin"])
    ok = bool(c0 and c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.20_horizon_shape",
        "observable": "surface dS vs Tr(K dC); CHM vs linear vs flat; stable n_occ",
        "one_d": d1,
        "three_d": d3,
        "C0_instrument": c0,
        "C1_rho": c1,
        "C2_Rshape": c2,
        "C3_beats_flat": c3,
        "C4_PRIMARY_beats_linear": c4,
        "all_gates": ok,
        "verdict": (
            "HORIZON_SHAPE_INSTRUMENT_REJECTED"
            if not c0
            else ("HORIZON_CHM_SELECTED" if ok else "HORIZON_CHM_NOT_UNIQUE")
        ),
        "not_claimed": [
            "eta = 1/4G",
            "Einstein",
            "foam",
            "Bloch",
            "de Sitter",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_20_horizon_shape.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
