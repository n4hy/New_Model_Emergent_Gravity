#!/usr/bin/env python3
"""M9.29 audit. N=10, source (4,4,4), σ=1.2, α=0.03.

Balls R=2 and cubes side=3. No import. Tries to REFUTE:
  control (ball export beats flat),
  cube export C2e,
  cube native C2n and C2x.

Writes ../data/m9_29_audit_shape.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SIG, ALPHA = 10, 1.2, 0.03
SRC = (4, 4, 4)


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


def S(block):
    z = np.clip(np.linalg.eigvalsh(block), CLIP, 1 - CLIP)
    return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))


def K(block):
    z, u = np.linalg.eigh(block)
    z = np.clip(z, CLIP, 1 - CLIP)
    return (u * np.log((1 - z) / z)) @ u.T


def score(kind, c0, c1, de, dc):
    ds, pe, pn, pf, pk = [], [], [], [], []
    if kind == "ball":
        lo, hi, rad = 2, N - 2, 2
        for cx in range(lo, hi):
            for cy in range(lo, hi):
                for cz in range(lo, hi):
                    sl, r2, eacc = [], [], 0.0
                    for x in range(N):
                        for y in range(N):
                            for z in range(N):
                                dx, dy, dz = x - cx, y - cy, z - cz
                                rr = dx * dx + dy * dy + dz * dz
                                if rr <= rad * rad:
                                    i = idx(x, y, z)
                                    sl.append(i)
                                    r2.append(rr)
                                    eacc += de[i]
                    sl = np.array(sl, dtype=int)
                    r2 = np.asarray(r2, float)
                    wexp = float(np.max(r2)) - r2
                    ds.append(S(c1[np.ix_(sl, sl)]) - S(c0[np.ix_(sl, sl)]))
                    pe.append(float(np.dot(wexp, de[sl])))
                    pn.append(pe[-1])
                    pf.append(float(eacc))
                    pk.append(float(np.sum(K(c0[np.ix_(sl, sl)]) * dc[np.ix_(sl, sl)])))
    else:
        lo, hi = 1, N - 1
        half2 = (1.5) ** 2
        for cx in range(lo, hi):
            for cy in range(lo, hi):
                for cz in range(lo, hi):
                    sl, rel = [], []
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                sl.append(idx(cx + dx, cy + dy, cz + dz))
                                rel.append((dx, dy, dz))
                    sl = np.array(sl, dtype=int)
                    r2 = np.array([a * a + b * b + c * c for a, b, c in rel], float)
                    wexp = float(np.max(r2)) - r2
                    wnat = np.array(
                        [(half2 - a * a) * (half2 - b * b) * (half2 - c * c) for a, b, c in rel],
                        float,
                    )
                    ds.append(S(c1[np.ix_(sl, sl)]) - S(c0[np.ix_(sl, sl)]))
                    pe.append(float(np.dot(wexp, de[sl])))
                    pn.append(float(np.dot(wnat, de[sl])))
                    pf.append(float(np.sum(de[sl])))
                    pk.append(float(np.sum(K(c0[np.ix_(sl, sl)]) * dc[np.ix_(sl, sl)])))
    ds = np.asarray(ds, float)
    re, rn, rf = rshape(ds, pe), rshape(ds, pn), rshape(ds, pf)
    rho_k = pearson(ds, pk)
    return {
        "n": int(len(ds)),
        "rho_Kvac": rho_k,
        "rho_export": pearson(ds, pe),
        "rho_native": pearson(ds, pn),
        "rho_flat": pearson(ds, pf),
        "R_export": re,
        "R_native": rn,
        "R_flat": rf,
        "C_vac": bool(abs(rho_k) > 0.95),
        "C2e": bool(re < rf),
        "C2n": bool(rn < rf),
        "C2x": bool(rn < re),
    }


def main() -> int:
    vol = N**3
    ham = np.zeros((vol, vol))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        j = idx(xx, yy, zz)
                        ham[i, j] = ham[j, i] = -1.0
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
    dc = c1 - c0
    ball = score("ball", c0, c1, de, dc)
    cube = score("cube", c0, c1, de, dc)
    payload = {
        "task": "m9.29_audit_shape",
        "ball": ball,
        "cube": cube,
        "verdicts": {
            "ball_control": "CONFIRMED" if (ball["C_vac"] and ball["C2e"]) else "REFUTED",
            "cube_C2e": "CONFIRMED" if cube["C2e"] else "REFUTED",
            "cube_C2n": "CONFIRMED" if cube["C2n"] else "REFUTED",
            "cube_C2x": "CONFIRMED" if cube["C2x"] else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_29_audit_shape.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    ok = bool(ball["C_vac"] and ball["C2e"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
