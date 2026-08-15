#!/usr/bin/env python3
"""M9.36: Gauss first-law constant κ = δS / M_enc.

Paper 45: well inside, δS tracks enclosed energy. This run
asks whether that ratio is one number, reusable across
source counts.

PRE-REGISTERED:
  N=12, R=3, 216 balls. H fixed. α=0.02.
  Config 1: one packet at (6,6,6), σ=1.0.
  Config 2: two packets at (5,6,6) and (7,6,6), σ=1.0.
  Well-inside: ball contains every source.
  κ = δS / P_flat on well-inside balls with |P_flat| > 1e-6.
  C_vac   |ρ(δS, Tr(K_vac ΔC))| > 0.95 on each config
  C_track |ρ(δS, P_flat)| > 0.95 on each well-inside set
  C_univ  PRIMARY. |med κ1 − med κ2| / mean|med| < 0.15
          and rel IQR of κ < 0.35 on each config
  C_pred  κ1 × P_flat^(2) predicts δS^(2): Pearson > 0.95
          on config-2 well-inside balls

Not claimed: 8πG, 1/4G, Einstein, de Sitter, MODELS.md.

Writes ../data/m9_36_kappa.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
RADIUS = 3
SIGMA = 1.0
ALPHA = 0.02


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
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


def raw_packet(uo, uu, n, src, sigma):
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
    return uo @ (uo.T @ env), uu @ (uu.T @ stag)


def orthonormalize(v1, v2):
    e1 = v1 / np.linalg.norm(v1)
    v2 = v2 - e1 * np.dot(e1, v2)
    n2 = np.linalg.norm(v2)
    if n2 < 1e-14:
        raise RuntimeError("packets linearly dependent")
    return e1, v2 / n2


def states(ham, n, sources, sigma, alpha):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T
    lefts, rights = [], []
    for src in sources:
        left, right = raw_packet(uo, uu, n, src, sigma)
        lefts.append(left)
        rights.append(right)
    if len(sources) == 1:
        lefts[0] = lefts[0] / np.linalg.norm(lefts[0])
        rights[0] = rights[0] / np.linalg.norm(rights[0])
    else:
        lefts[0], lefts[1] = orthonormalize(lefts[0], lefts[1])
        rights[0], rights[1] = orthonormalize(rights[0], rights[1])
    corr = c0.copy()
    for left, right in zip(lefts, rights):
        corr = corr + alpha * (np.outer(right, right) - np.outer(left, left))
    return c0, 0.5 * (corr + corr.T)


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr, sl):
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def in_ball(src, center, radius):
    return sum((src[k] - center[k]) ** 2 for k in range(3)) <= radius * radius


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def kappa_stats(ds, pflat, inside):
    mask = inside & (np.abs(pflat) > 1e-6)
    kap = ds[mask] / pflat[mask]
    med = float(np.median(kap))
    iqr = float(np.percentile(kap, 75) - np.percentile(kap, 25))
    rel = float(iqr / abs(med)) if med != 0.0 else None
    return {
        "n": int(mask.sum()),
        "median": med,
        "rel_iqr": rel,
        "rho": pearson(ds[mask], pflat[mask]),
        "iqr_pass": bool(rel is not None and rel < 0.35),
        "track_pass": bool(abs(pearson(ds[mask], pflat[mask])) > 0.95),
    }


def measure(ham, sources):
    c0, c1 = states(ham, N, sources, SIGMA, ALPHA)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds, pflat, pk, inside = [], [], [], []
    for cx, cy, cz in centers:
        sl = np.array(
            [
                idx(x, y, z)
                for x in range(N)
                for y in range(N)
                for z in range(N)
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2max
            ],
            dtype=int,
        )
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pk.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
        s_f = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        s_f += de[idx(x, y, z)]
        pflat.append(s_f)
        inside.append(all(in_ball(s, (cx, cy, cz), RADIUS) for s in sources))
    ds, pflat, pk = map(np.asarray, (ds, pflat, pk))
    inside = np.asarray(inside, bool)
    return {
        "rho_Kvac": pearson(ds, pk),
        "kappa": kappa_stats(ds, pflat, inside),
        "ds": ds,
        "pflat": pflat,
        "inside": inside,
    }


def main() -> int:
    ham = hop_H(N)
    one = measure(ham, [(6, 6, 6)])
    two = measure(ham, [(5, 6, 6), (7, 6, 6)])
    k1, k2 = one["kappa"], two["kappa"]
    mean_abs = 0.5 * (abs(k1["median"]) + abs(k2["median"]))
    rel_med = abs(k1["median"] - k2["median"]) / mean_abs if mean_abs else None
    c_vac = bool(abs(one["rho_Kvac"]) > 0.95 and abs(two["rho_Kvac"]) > 0.95)
    c_track = bool(k1["track_pass"] and k2["track_pass"])
    c_univ = bool(
        rel_med is not None
        and rel_med < 0.15
        and k1["iqr_pass"]
        and k2["iqr_pass"]
    )
    pred = k1["median"] * two["pflat"][two["inside"]]
    rho_pred = pearson(two["ds"][two["inside"]], pred)
    c_pred = bool(abs(rho_pred) > 0.95)
    ok = bool(c_vac and c_track and c_univ and c_pred)
    payload = {
        "task": "m9.36_kappa",
        "one": {"rho_Kvac": one["rho_Kvac"], "kappa": k1},
        "two": {"rho_Kvac": two["rho_Kvac"], "kappa": k2},
        "rel_median": rel_med,
        "rho_pred_2_from_k1": rho_pred,
        "C_vac": c_vac,
        "C_track": c_track,
        "C_univ_PRIMARY": c_univ,
        "C_pred": c_pred,
        "all_gates": ok,
        "verdict": "KAPPA_UNIVERSAL" if ok else "KAPPA_NOT_UNIVERSAL",
        "not_claimed": ["8pi G", "1/4G", "Einstein equation", "de Sitter", "MODELS.md"],
    }
    # drop arrays
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_36_kappa.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
