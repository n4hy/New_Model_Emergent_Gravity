#!/usr/bin/env python3
"""M9.48 audit. N=10, src (4,5,5), σ=0.9, α=0.03.

Tries to REFUTE C_sub.

Writes ../data/m9_48_audit_subtract.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SRC, SIG, ALPHA = 10, (4, 5, 5), 0.9, 0.03
RADII = (2, 3, 4)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def main() -> int:
    ham = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        ham[i, idx(xx, yy, zz)] = ham[idx(xx, yy, zz), i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros(N**3)
    stag = np.zeros(N**3)
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
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    c0 = uo @ uo.T
    c1 = 0.5 * (
        (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left)))
        + (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))).T
    )
    e0 = np.sum(ham * c0, axis=1)
    de = np.sum(ham * c1, axis=1) - e0
    e1 = e0 + de
    ds, pvac, pde, pe = [], [], [], []
    for rad in RADII:
        sl = np.array(
            [
                idx(x, y, z)
                for x in range(N)
                for y in range(N)
                for z in range(N)
                if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= rad * rad
            ],
            dtype=int,
        )
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pvac.append(float(np.sum(e0[sl])))
        pde.append(float(np.sum(de[sl])))
        pe.append(float(np.sum(e1[sl])))
    ds, pvac, pde, pe = map(np.asarray, (ds, pvac, pde, pe))
    rho_de, rho_e = pearson(ds, pde), pearson(ds, pe)
    c_sign = bool(all(pvac < 0.0) and all(pde > 0.0))
    c_sub = bool(abs(rho_de) > 0.95 and abs(rho_e) < abs(rho_de))
    payload = {
        "task": "m9.48_audit_subtract",
        "P_vac": pvac.tolist(),
        "P_de": pde.tolist(),
        "P_e": pe.tolist(),
        "rho_de": rho_de,
        "rho_e": rho_e,
        "scale_Rmax": abs(float(pvac[-1] / pde[-1])),
        "C_sign": c_sign,
        "C_sub": c_sub,
        "verdicts": {
            "C_sign": "CONFIRMED" if c_sign else "REFUTED",
            "C_sub": "CONFIRMED" if c_sub else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_48_audit_subtract.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_sign and c_sub) else 1


if __name__ == "__main__":
    raise SystemExit(main())
