#!/usr/bin/env python3
"""M9.33 audit. N=10, R=2, packet (4,5,5), σ=1.2, α=0.03.

Own cut-bond area. Tries to REFUTE C_eta and independence.

Writes ../data/m9_33_audit_state.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, ALPHA = 10, 2, 1.2, 0.03
SRC = (4, 5, 5)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


def main() -> int:
    vol = N**3
    ham = np.zeros((vol, vol))
    bonds = []
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        j = idx(xx, yy, zz)
                        ham[i, j] = ham[j, i] = -1.0
                        bonds.append((i, j))
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros(vol)
    stag = np.zeros(vol)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (SIG * SIG))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    c1 = c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))
    c1 = 0.5 * (c1 + c1.T)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    r2max = RADIUS * RADIUS

    def S(c, sl):
        z = np.clip(np.linalg.eigvalsh(c[np.ix_(sl, sl)]), CLIP, 1 - CLIP)
        return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

    def abond(c, inside):
        acc = 0.0
        for i, j in bonds:
            if (i in inside) != (j in inside):
                acc += abs(c[i, j])
        return acc

    ds, da, pc = [], [], []
    for cx in range(RADIUS, N - RADIUS):
        for cy in range(RADIUS, N - RADIUS):
            for cz in range(RADIUS, N - RADIUS):
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
                inside = set(int(i) for i in sl)
                ds.append(S(c1, sl) - S(c0, sl))
                da.append(abond(c1, inside) - abond(c0, inside))
                s = 0.0
                for x in range(N):
                    for y in range(N):
                        for z in range(N):
                            rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                            if rr <= r2max:
                                s += (r2max - rr) * de[idx(x, y, z)]
                pc.append(s)
    ds, da, pc = map(np.asarray, (ds, da, pc))
    mask = np.abs(da) > 1e-8
    if int(mask.sum()) == 0:
        rel, med = None, None
    else:
        eta = ds[mask] / da[mask]
        med = float(np.median(eta))
        iqr = float(np.percentile(eta, 75) - np.percentile(eta, 25))
        rel = float(iqr / abs(med)) if med != 0.0 else None
    rap = pearson(da, pc)
    ras = pearson(da, ds)
    c_eta = bool(rel is not None and rel < 0.35)
    c_ind = bool(np.isfinite(rap) and abs(rap) < 0.90)
    payload = {
        "task": "m9.33_audit_state",
        "n": int(len(ds)),
        "rho_S_bond": ras,
        "rho_bond_P": rap,
        "eta_median": med,
        "eta_rel_iqr": rel,
        "max_abs_dA": float(np.max(np.abs(da))),
        "C_indepP": c_ind,
        "C_eta": c_eta,
        "verdicts": {
            "C_indepP": "CONFIRMED" if c_ind else "REFUTED",
            "C_eta": "CONFIRMED" if c_eta else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_33_audit_state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_ind and c_eta) else 1


if __name__ == "__main__":
    raise SystemExit(main())
