#!/usr/bin/env python3
"""M9.51 audit. N=10, src (4,5,5), σ=0.9, α ∈ {0.015, 0.045}.

Own P_flat Gauss. Tries to REFUTE C_hold.

Writes ../data/m9_51_audit_energy.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N, SRC, SIG = 10, (4, 5, 5), 0.9
ALPHAS = (0.015, 0.045)
RADII = (2, 3, 4)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def slope_of(radii, accs):
    lr = np.log(np.asarray(radii, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([lr, np.ones(len(radii))]), la, rcond=None
    )
    return float(coef[0])


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
    e0 = np.sum(ham * c0, axis=1)
    dC = np.outer(right, right) - np.outer(left, left)
    sls = []
    for rad in RADII:
        sls.append(
            np.array(
                [
                    idx(x, y, z)
                    for x in range(N)
                    for y in range(N)
                    for z in range(N)
                    if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= rad * rad
                ],
                dtype=int,
            )
        )
    star_slopes = []
    for alpha in ALPHAS:
        c1 = 0.5 * ((c0 + alpha * dC) + (c0 + alpha * dC).T)
        de = np.sum(ham * c1, axis=1) - e0
        p = np.array([float(np.sum(de[sl])) for sl in sls])
        m = float(np.sum(de))
        encl = p / m
        fit_r = [r for r, e in zip(RADII, encl) if e > 0.95]
        a = -p / np.asarray(RADII, float) ** 2
        if len(fit_r) >= 2:
            star_slopes.append(slope_of(fit_r, [a[RADII.index(r)] for r in fit_r]))
        else:
            star_slopes.append(None)

    hamu = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % N, (y + d[1]) % N, (z + d[2]) % N)
                    hamu[i, j] = hamu[j, i] = -1.0
    evu, vecu = np.linalg.eigh(hamu)
    il, ir = int(np.argmin(evu)), int(np.argmax(evu))
    occu = evu < 0.0
    c0u = vecu[:, occu] @ vecu[:, occu].T
    e0u = np.sum(hamu * c0u, axis=1)
    dCu = np.outer(vecu[:, ir], vecu[:, ir]) - np.outer(vecu[:, il], vecu[:, il])
    slp = []
    for rad in RADII:
        sl = []
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    dx = min((x - sx) % N, (sx - x) % N)
                    dy = min((y - sy) % N, (sy - y) % N)
                    dz = min((z - sz) % N, (sz - z) % N)
                    if dx * dx + dy * dy + dz * dz <= rad * rad:
                        sl.append(idx(x, y, z))
        slp.append(np.array(sl, dtype=int))
    sea_slopes = []
    for alpha in ALPHAS:
        c1u = 0.5 * ((c0u + alpha * dCu) + (c0u + alpha * dCu).T)
        de = np.sum(hamu * c1u, axis=1) - e0u
        p = np.array([float(np.sum(de[sl])) for sl in slp])
        a = -p / np.asarray(RADII, float) ** 2
        sea_slopes.append(slope_of(RADII, a))

    ss = [s for s in star_slopes if s is not None]
    c_hold = bool(ss and (max(ss) - min(ss)) < 0.10 and (max(sea_slopes) - min(sea_slopes)) < 0.10)
    c_star = bool(ss and all(abs(s + 2.0) < 0.20 for s in ss))
    c_sea = bool(all(abs(s - 1.0) < abs(s + 2.0) for s in sea_slopes))
    payload = {
        "task": "m9.51_audit_energy",
        "star_slopes": star_slopes,
        "sea_slopes": sea_slopes,
        "C_star": c_star,
        "C_sea": c_sea,
        "C_hold": c_hold,
        "verdicts": {
            "C_star": "CONFIRMED" if c_star else "REFUTED",
            "C_sea": "CONFIRMED" if c_sea else "REFUTED",
            "C_hold": "CONFIRMED" if c_hold else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_51_audit_energy.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_hold and c_star and c_sea) else 1


if __name__ == "__main__":
    raise SystemExit(main())
