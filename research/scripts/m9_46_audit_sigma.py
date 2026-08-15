#!/usr/bin/env python3
"""M9.46 audit. N=10, src (4,5,5), α=0.03, σ ∈ {1.0, 4.0}, own PBC sea.

Tries to REFUTE C_in and C_mono.

Writes ../data/m9_46_audit_sigma.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SRC, ALPHA = 10, (4, 5, 5), 0.03
SIGMAS = (1.0, 4.0)
RADII = (2, 3, 4)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


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
    sx, sy, sz = SRC
    slopes = []
    inward = True
    for sig in SIGMAS:
        env = np.zeros(N**3)
        stag = np.zeros(N**3)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                    env[i] = np.exp(-0.5 * rr / (sig * sig))
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
        de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
        ds, pf = [], []
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
            pf.append(float(np.sum(de[sl])))
        ds, pf = np.asarray(ds), np.asarray(pf)
        mglob = float(np.sum(de))
        encl = pf / mglob
        k_idx = next((i for i, e in enumerate(encl) if e > 0.95), len(RADII) - 1)
        kap = ds[k_idx] / pf[k_idx]
        accs = -(ds / kap) / np.asarray(RADII, float) ** 2
        if not all(a < 0.0 for a in accs):
            inward = False
        slopes.append(slope_of(RADII, accs))

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
    c1u = 0.5 * (
        (c0u + ALPHA * (np.outer(vecu[:, ir], vecu[:, ir]) - np.outer(vecu[:, il], vecu[:, il])))
        + (c0u + ALPHA * (np.outer(vecu[:, ir], vecu[:, ir]) - np.outer(vecu[:, il], vecu[:, il]))).T
    )
    deu = np.sum(hamu * c1u, axis=1) - np.sum(hamu * c0u, axis=1)
    dsu, pfu = [], []
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
        sl = np.array(sl, dtype=int)
        dsu.append(peschel_s(c1u, sl) - peschel_s(c0u, sl))
        pfu.append(float(np.sum(deu[sl])))
    dsu, pfu = np.asarray(dsu), np.asarray(pfu)
    kapu = dsu[0] / pfu[0]
    au = -(dsu / kapu) / np.asarray(RADII, float) ** 2
    if not all(a < 0.0 for a in au):
        inward = False
    sea_slope = slope_of(RADII, au)
    c_in = bool(inward)
    c_mono = bool(slopes[0] <= slopes[1] + 1e-12)
    payload = {
        "task": "m9.46_audit_sigma",
        "slopes": slopes,
        "sea_slope": sea_slope,
        "C_in": c_in,
        "C_mono": c_mono,
        "verdicts": {
            "C_in": "CONFIRMED" if c_in else "REFUTED",
            "C_mono": "CONFIRMED" if c_mono else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_46_audit_sigma.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_in and c_mono) else 1


if __name__ == "__main__":
    raise SystemExit(main())
