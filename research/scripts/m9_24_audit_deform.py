#!/usr/bin/env python3
"""Region-deform audit. N=14. No solver import.

Balls R=2,3,4 and cubes L=3,4,5. Fit α,β on balls; cubes must
have larger RMS residual (C2).

Writes ../data/m9_24_audit_deform.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 14


def ix(x, y, z, n=N):
    return (x * n + y) * n + z


def main() -> int:
    ham = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = ix(x, y, z)
                if x + 1 < N:
                    ham[i, ix(x + 1, y, z)] = ham[ix(x + 1, y, z), i] = -1.0
                if y + 1 < N:
                    ham[i, ix(x, y + 1, z)] = ham[ix(x, y + 1, z), i] = -1.0
                if z + 1 < N:
                    ham[i, ix(x, y, z + 1)] = ham[ix(x, y, z + 1), i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    cfull = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    cen = N // 2

    def S_of(coords):
        sl = np.array([ix(*p) for p in coords])
        w = np.clip(np.linalg.eigvalsh(cfull[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
        return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))

    def area(coords):
        inside = set(coords)
        cuts = 0
        for x, y, z in coords:
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                if (x + d[0], y + d[1], z + d[2]) not in inside:
                    cuts += 1
        return cuts

    def ball(r):
        return [
            (x, y, z)
            for x in range(N)
            for y in range(N)
            for z in range(N)
            if (x - cen) ** 2 + (y - cen) ** 2 + (z - cen) ** 2 <= r * r
        ]

    def cube(s):
        lo = cen - s // 2
        return [
            (x, y, z)
            for x in range(lo, lo + s)
            for y in range(lo, lo + s)
            for z in range(lo, lo + s)
            if 0 <= x < N and 0 <= y < N and 0 <= z < N
        ]

    balls = [(area(ball(r)), S_of(ball(r))) for r in (2, 3, 4)]
    cubes = [(area(cube(s)), S_of(cube(s))) for s in (3, 4, 5)]
    ab, sb = zip(*balls)
    ac, sc = zip(*cubes)
    mat = np.column_stack([ab, np.ones(len(ab))])
    coef, _, _, _ = np.linalg.lstsq(mat, sb, rcond=None)
    pred_b = mat @ coef
    rms_b = float(np.sqrt(np.mean((np.array(sb) - pred_b) ** 2)))
    pred_c = np.array(ac) * coef[0] + coef[1]
    rms_c = float(np.sqrt(np.mean((np.array(sc) - pred_c) ** 2)))
    c2 = bool(rms_c > rms_b)
    payload = {
        "task": "m9.24_audit_deform",
        "method": "N=14; balls R=2,3,4; cubes L=3,4,5; no import",
        "alpha": float(coef[0]),
        "rms_ball": rms_b,
        "rms_cube": rms_c,
        "balls": [{"A": a, "S": s} for a, s in balls],
        "cubes": [{"A": a, "S": s} for a, s in cubes],
        "C2": c2,
        "verdicts": {"C2": "CONFIRMED" if c2 else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_24_audit_deform.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
