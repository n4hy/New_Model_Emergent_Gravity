#!/usr/bin/env python3
"""A1 audit of the 3+1D diamond-waist area coefficient.

Independent of m9_14_A1_diamond_4d.py (no solver import).
Different grid and radii: N=14, R in {2,3,4}. Same C2.

Writes ../data/m9_14_audit_A1.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N = 14
RADII = (2, 3, 4)
EPS = 1e-12
REL_THRESH = 0.20


def idx(x: int, y: int, z: int, n: int) -> int:
    return (x * n + y) * n + z


def alpha_of(n: int, mass: float, radii: tuple[int, ...]) -> float:
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
    areas = []
    ents = []
    for radius in radii:
        r2 = radius * radius
        coords = []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= r2:
                        coords.append((x, y, z))
        inside = set(coords)
        cuts = 0
        for x, y, z in coords:
            for d in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ):
                if (x + d[0], y + d[1], z + d[2]) not in inside:
                    cuts += 1
        sl = np.array([idx(x, y, z, n) for x, y, z in coords])
        w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), EPS, 1.0 - EPS)
        ent = float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))
        areas.append(float(cuts))
        ents.append(ent)
    a = np.array(areas)
    s = np.array(ents)
    mat = np.column_stack([a, np.ones(len(a))])
    coef, _, _, _ = np.linalg.lstsq(mat, s, rcond=None)
    return float(coef[0])


def main() -> int:
    a0 = alpha_of(N, 0.0, RADII)
    a_uv = alpha_of(N, 0.06, RADII)  # m R_max = 0.24
    rel = abs(a_uv - a0) / abs(a0) if a0 != 0.0 else None
    c1 = bool(a0 > 0.0)
    c2 = bool(rel is not None and rel < REL_THRESH)
    payload = {
        "task": "m9.14_audit_A1",
        "method": "N=14, R=2,3,4; no solver import",
        "alpha0": a0,
        "alpha_uv": a_uv,
        "rel": rel,
        "C1": c1,
        "C2": c2,
        "verdicts": {"C2": "CONFIRMED" if (c1 and c2) else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_14_audit_A1.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c1 and c2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
