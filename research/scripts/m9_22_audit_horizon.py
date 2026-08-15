#!/usr/bin/env python3
"""Half-fill R=5 C4 audit. No solver import.

1d N=160 L=20. 3d N=16 R=5, 5/bin, ε=0.012, lowest V/2 occupied.

Writes ../data/m9_22_audit_horizon.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
EPS_H = 0.012
BINS = ((3.75, 4.20), (4.20, 4.60), (4.60, 5.01))
PER_BIN = 5


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
    nocc = ham.shape[0] // 2
    full = vecs[:, :nocc] @ vecs[:, :nocc].T
    ca = full[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, CLIP, 1.0 - CLIP)
    ent = float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))
    k = (u * np.log((1.0 - w) / w)) @ u.T
    return ent, 0.5 * (k + k.T), ca, nocc


def fit_w(k, pairs, weights):
    knn = np.array([k[i, j] for i, j in pairs])
    mat = np.column_stack([weights, np.ones(len(weights))])
    coef, _, _, _ = np.linalg.lstsq(mat, knn, rcond=None)
    out = np.zeros_like(k)
    for (i, j), ww in zip(pairs, weights):
        out[i, j] = out[j, i] = coef[0] * ww + coef[1]
    return out


def kflat(k, pairs):
    fill = float(np.mean([k[i, j] for i, j in pairs]))
    out = np.zeros_like(k)
    for i, j in pairs:
        out[i, j] = out[j, i] = fill
    return out


def audit_1d() -> dict:
    n, ell = 160, 20
    ham0 = np.zeros((n, n))
    for i in range(n - 1):
        ham0[i, i + 1] = ham0[i + 1, i] = -1.0
    mid = n // 2
    sl = np.arange(mid - ell // 2, mid + ell // 2)
    s0, k0, c0, n0 = peschel(ham0, sl)
    x = np.arange(ell) - (ell - 1) / 2.0
    kept, ds, dcs, wc = [], [], [], []
    for i in range(ell - 1):
        xm = 0.5 * (x[i] + x[i + 1])
        if abs(xm) < ell / 4.0:
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
        wc.append((ell / 2.0) ** 2 - xm**2)
    kc = fit_w(k0, kept, np.array(wc))
    pc = [float(np.trace(kc @ d)) for d in dcs]
    rho = pearson(ds, pc)
    return {"n": int(len(ds)), "rho_chm": rho, "C0": bool(abs(rho) > 0.70)}


def audit_3d() -> dict:
    n, radius = 16, 5

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
    coords = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cen) ** 2 + (y - cen) ** 2 + (z - cen) ** 2 <= radius * radius:
                    coords.append((x, y, z))
    pos = {p: i for i, p in enumerate(coords)}
    sl = np.array([ix(*p) for p in coords])
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
                    buckets[b].append((x, y, z, d, i, j, rm, nbr))
                    break
    picked = []
    for bucket in buckets:
        bucket.sort()
        picked.extend(bucket[:PER_BIN])
    s0, k0, c0, n0 = peschel(ham0, sl)
    kept, ds, dcs, rmid = [], [], [], []
    n_flip = 0
    for x, y, z, d, i, j, rm, nbr in picked:
        a, b = ix(x, y, z), ix(*nbr)
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
    rmid = np.array(rmid)
    kc = fit_w(k0, kept, radius * radius - rmid**2)
    kl = fit_w(k0, kept, radius - rmid)
    kf = kflat(k0, kept)
    pc = [float(np.trace(kc @ d)) for d in dcs]
    pl = [float(np.trace(kl @ d)) for d in dcs]
    pf = [float(np.trace(kf @ d)) for d in dcs]
    rc, rl, rf = rshape(ds, pc), rshape(ds, pl), rshape(ds, pf)
    rho = pearson(ds, pc)
    return {
        "n": int(len(ds)),
        "n_flip": int(n_flip),
        "rho_chm": rho,
        "R_chm": rc,
        "R_lin": rl,
        "R_flat": rf,
        "C1": bool(abs(rho) > 0.60),
        "C2": bool(rc < 0.70),
        "C3": bool(rc < rf),
        "C4": bool(rc < rl),
    }


def main() -> int:
    d1 = audit_1d()
    d3 = audit_3d()
    ok = bool(d1["C0"] and d3["C1"] and d3["C2"] and d3["C3"] and d3["C4"])
    payload = {
        "task": "m9.22_audit_horizon",
        "method": "half-fill; N=16 R=5, 5/bin; no solver import",
        "one_d": d1,
        "three_d": d3,
        "verdicts": {"C4": "CONFIRMED" if ok else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_22_audit_horizon.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
