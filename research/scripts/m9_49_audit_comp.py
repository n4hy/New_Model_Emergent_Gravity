#!/usr/bin/env python3
"""M9.49 audit. N=10, src (4,5,5), σ=0.9, α=0.03, R=3,4.

Tries to REFUTE C_comp (or confirm the plus sign).

Writes ../data/m9_49_audit_comp.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SRC, SIG, ALPHA = 10, (4, 5, 5), 0.9, 0.03
RADII = (3, 4)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


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
    rows = []
    c_in = True
    c_comp = True
    c_pure = True
    for rad in RADII:
        inside, outside = [], []
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= rad * rad:
                        inside.append(i)
                    else:
                        outside.append(i)
        sl_b, sl_c = np.array(inside, dtype=int), np.array(outside, dtype=int)
        s0b, s0c = peschel_s(c0, sl_b), peschel_s(c0, sl_c)
        s1b, s1c = peschel_s(c1, sl_b), peschel_s(c1, sl_c)
        dsb, dsc = s1b - s0b, s1c - s0c
        if s0b and abs(s0b - s0c) / s0b >= 0.02:
            c_pure = False
        if dsb <= 0.0:
            c_in = False
        if dsc >= 0.0:
            c_comp = False
        rows.append({"R": rad, "dS_B": dsb, "dS_Bc": dsc, "S0_B": s0b, "S0_Bc": s0c})
    payload = {
        "task": "m9.49_audit_comp",
        "rows": rows,
        "C_pure": c_pure,
        "C_in": c_in,
        "C_comp": c_comp,
        "verdicts": {
            "C_in": "CONFIRMED" if c_in else "REFUTED",
            "C_comp": "CONFIRMED" if c_comp else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_49_audit_comp.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c_in else 1


if __name__ == "__main__":
    raise SystemExit(main())
