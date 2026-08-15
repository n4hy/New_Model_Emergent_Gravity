#!/usr/bin/env python3
"""M9.30 audit. N=10, R=2, Φ at (4,5,5), σ=1.5, ε=0.03.

No import. Tries to REFUTE C_area and C_eta.

Writes ../data/m9_30_audit_area.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, EPS = 10, 2, 1.5, 0.03
SRC = (4, 5, 5)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


def rshape(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    m = np.column_stack([x, np.ones(len(x))])
    c, _, _, _ = np.linalg.lstsq(m, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    return float("nan") if den == 0 else float(np.linalg.norm(y - m @ c) / den)


def main() -> int:
    vol = N**3
    phi = np.zeros(vol)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                phi[idx(x, y, z)] = np.exp(-0.5 * rr / (SIG * SIG))

    def ham_of(eps):
        ham = np.zeros((vol, vol))
        tm = {}
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        xx, yy, zz = x + d[0], y + d[1], z + d[2]
                        if xx < N and yy < N and zz < N:
                            j = idx(xx, yy, zz)
                            t = -(1.0 + eps * 0.5 * (phi[i] + phi[j]))
                            ham[i, j] = ham[j, i] = t
                            tm[(min(i, j), max(i, j))] = t
        return ham, tm

    def occ(ham):
        ev, vecs = np.linalg.eigh(ham)
        fill = ev < 0.0
        return vecs[:, fill] @ vecs[:, fill].T

    def S(c, sl):
        z = np.clip(np.linalg.eigvalsh(c[np.ix_(sl, sl)]), CLIP, 1 - CLIP)
        return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

    def K(c, sl):
        b = c[np.ix_(sl, sl)]
        z, u = np.linalg.eigh(b)
        z = np.clip(z, CLIP, 1 - CLIP)
        return (u * np.log((1 - z) / z)) @ u.T

    def area(sl, tm):
        inside = set(int(i) for i in sl)
        acc = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    if i not in inside:
                        continue
                    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        xx, yy, zz = x + d[0], y + d[1], z + d[2]
                        if not (0 <= xx < N and 0 <= yy < N and 0 <= zz < N):
                            acc += 1.0
                            continue
                        j = idx(xx, yy, zz)
                        if j not in inside:
                            acc += 1.0 / abs(tm[(min(i, j), max(i, j))])
        return acc

    h0, t0 = ham_of(0.0)
    h1, t1 = ham_of(EPS)
    c0, c1 = occ(h0), occ(h1)
    dc = c1 - c0
    r2max = RADIUS * RADIUS
    ds, da, pk = [], [], []
    for cx in range(RADIUS, N - RADIUS):
        for cy in range(RADIUS, N - RADIUS):
            for cz in range(RADIUS, N - RADIUS):
                sl = []
                for x in range(N):
                    for y in range(N):
                        for z in range(N):
                            if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2max:
                                sl.append(idx(x, y, z))
                sl = np.array(sl, dtype=int)
                ds.append(S(c1, sl) - S(c0, sl))
                da.append(area(sl, t1) - area(sl, t0))
                pk.append(float(np.sum(K(c0, sl) * dc[np.ix_(sl, sl)])))
    ds, da = np.asarray(ds, float), np.asarray(da, float)
    mask = np.abs(da) > 1e-4
    eta = ds[mask] / da[mask]
    med = float(np.median(eta))
    iqr = float(np.percentile(eta, 75) - np.percentile(eta, 25))
    rel = float(iqr / abs(med)) if med != 0.0 else float("inf")
    rho_a = pearson(ds, da)
    c_area = bool(abs(rho_a) > 0.80)
    c_eta = bool(rel < 0.35)
    c_vac = bool(abs(pearson(ds, pk)) > 0.95)
    payload = {
        "task": "m9.30_audit_area",
        "n": int(len(ds)),
        "rho_Kvac": pearson(ds, pk),
        "rho_area": rho_a,
        "R_area": rshape(ds, da),
        "eta_median": med,
        "eta_rel_iqr": rel,
        "C_vac": c_vac,
        "C_area": c_area,
        "C_eta": c_eta,
        "verdicts": {
            "C_vac": "CONFIRMED" if c_vac else "REFUTED",
            "C_area": "CONFIRMED" if c_area else "REFUTED",
            "C_eta": "CONFIRMED" if c_eta else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_30_audit_area.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_vac and c_area and c_eta) else 1


if __name__ == "__main__":
    raise SystemExit(main())
