#!/usr/bin/env python3
"""A2 audit of the 4d-continuum diamond waist.

Independent of m9_13_A2_diamond_4d.py (no solver import). Different
grid and radius: N=14, R=4, masses giving m L in {0, 3, 6} with
L = 2 R = 8. Same C2: R(m)/R(0) < 2.

Writes ../data/m9_13_audit_A2.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N = 14
RADIUS = 4
THRESH = 2.0
EPS = 1e-12
# L = 8; m L in {0, 3, 6}
MASSES = [0.0, 0.375, 0.75]


def idx(x: int, y: int, z: int, n: int) -> int:
    return (x * n + y) * n + z


def diamond_R(n: int, radius: int, mass: float) -> float:
    vol = n**3
    ham = np.zeros((vol, vol), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                ham[i, i] = mass * (1.0 if (x + y + z) % 2 == 0 else -1.0)
                if x + 1 < n:
                    j = idx(x + 1, y, z, n)
                    ham[i, j] = ham[j, i] = -1.0
                if y + 1 < n:
                    j = idx(x, y + 1, z, n)
                    ham[i, j] = ham[j, i] = -1.0
                if z + 1 < n:
                    j = idx(x, y, z + 1, n)
                    ham[i, j] = ham[j, i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    corr = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    c = n // 2
    r2 = radius * radius
    sl = []
    coords = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= r2:
                    sl.append(idx(x, y, z, n))
                    coords.append((x, y, z))
    sl = np.array(sl)
    pos = {pt: i for i, pt in enumerate(coords)}
    ca = corr[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, EPS, 1.0 - EPS)
    k = (u * np.log((1.0 - w) / w)) @ u.T
    loc = np.zeros_like(k)
    for i, (x, y, z) in enumerate(coords):
        loc[i, i] = k[i, i]
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            j = pos.get((x + d[0], y + d[1], z + d[2]))
            if j is not None:
                loc[i, j] = k[i, j]
    return float(np.linalg.norm(k - loc, "fro") / np.linalg.norm(k, "fro"))


def main() -> int:
    values = [diamond_R(N, RADIUS, m) for m in MASSES]
    r0, r1, r2 = values
    ratio1 = r1 / r0
    ratio2 = r2 / r0
    c1 = bool(0.0 < r0 < 1.0)
    c2 = bool(ratio1 < THRESH and ratio2 < THRESH)
    payload = {
        "task": "m9.13_audit_A2",
        "method": "3+1D ball waist N=14 R=4; no solver import",
        "n_grid": N,
        "R_ball": RADIUS,
        "L_diam": 2 * RADIUS,
        "R0": r0,
        "R_mL3": r1,
        "R_mL6": r2,
        "ratio_mL3": ratio1,
        "ratio_mL6": ratio2,
        "C1": c1,
        "C2": c2,
        "verdicts": {"C2_diamond": "CONFIRMED" if c2 else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_13_audit_A2.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c1 and c2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
