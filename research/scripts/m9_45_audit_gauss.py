#!/usr/bin/env python3
"""M9.45 audit. N=10, star σ=0.9 α=0.03 src (4,5,5); sea α=0.03.

Own first-law Gauss a(R). Tries to REFUTE C_star and C_sea.

Writes ../data/m9_45_audit_gauss.json
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


def slope_of(radii, accs):
    lr = np.log(np.asarray(radii, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([lr, np.ones(len(radii))]), la, rcond=None
    )
    return float(coef[0])


def main() -> int:
    # star
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
    r_enc = next(r for r, e in zip(RADII, encl) if e > 0.95)
    kap = float(ds[RADII.index(r_enc)] / pf[RADII.index(r_enc)])
    a_s = -(ds / kap) / np.asarray(RADII, float) ** 2
    star_rs = [r for r, e in zip(RADII, encl) if e > 0.95]
    slope_star = slope_of(star_rs, [a_s[RADII.index(r)] for r in star_rs])
    c_star = bool(len(star_rs) >= 2 and abs(slope_star + 2.0) < 0.20)

    # sea
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
    kapu = float(dsu[0] / pfu[0])
    a_u = -(dsu / kapu) / np.asarray(RADII, float) ** 2
    slope_sea = slope_of(RADII, a_u)
    c_sea = bool(abs(slope_sea - 1.0) < 0.40 and abs(slope_sea - 1.0) < abs(slope_sea + 2.0))
    payload = {
        "task": "m9.45_audit_gauss",
        "star_slope": slope_star,
        "star_R": star_rs,
        "sea_slope": slope_sea,
        "C_star": c_star,
        "C_sea": c_sea,
        "verdicts": {
            "C_star": "CONFIRMED" if c_star else "REFUTED",
            "C_sea": "CONFIRMED" if c_sea else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_45_audit_gauss.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_star and c_sea) else 1


if __name__ == "__main__":
    raise SystemExit(main())
