#!/usr/bin/env python3
"""M9.12 A2: 3d staggered-mass free fermion (lattice Dirac).

Local ansatz: on-site + 6 nearest neighbours. Same C2 as 1d/2d.

PRE-REGISTERED:
  C2 PRIMARY: R(m)/R(0) < 2 for 0 < m L ≤ 8.

Writes ../data/m9_12_A2_dirac_3d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N = 10
L = 6
THRESH = 2.0
EPS = 1e-12


def idx(x, y, z, n):
    return (x * n + y) * n + z


def staggered_H_3d(n: int, mass: float) -> np.ndarray:
    vol = n**3
    H = np.zeros((vol, vol), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                H[i, i] = mass * (1.0 if (x + y + z) % 2 == 0 else -1.0)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        H[i, j] = H[j, i] = -1.0
    return H


def local_mask(n_reg: int) -> np.ndarray:
    vol = n_reg**3
    mask = np.zeros((vol, vol), dtype=bool)
    for x in range(n_reg):
        for y in range(n_reg):
            for z in range(n_reg):
                i = idx(x, y, z, n_reg)
                mask[i, i] = True
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if 0 <= xx < n_reg and 0 <= yy < n_reg and 0 <= zz < n_reg:
                        mask[i, idx(xx, yy, zz, n_reg)] = True
    return mask


def main() -> int:
    mid = N // 2
    sl = []
    lo = mid - L // 2
    for x in range(lo, lo + L):
        for y in range(lo, lo + L):
            for z in range(lo, lo + L):
                sl.append(idx(x, y, z, N))
    sl = np.array(sl)
    loc = local_mask(L)
    masses = [0.0, 0.2, 0.5, 1.0]
    rows = []
    R0 = None
    for m in masses:
        H = staggered_H_3d(N, m)
        ev, vecs = np.linalg.eigh(H)
        C = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
        CA = C[np.ix_(sl, sl)]
        w, u = np.linalg.eigh(CA)
        w = np.clip(w, EPS, 1.0 - EPS)
        K = (u * np.log((1.0 - w) / w)) @ u.T
        locK = np.where(loc, K, 0.0)
        Rv = float(np.linalg.norm(K - locK, "fro") / np.linalg.norm(K, "fro"))
        if m == 0.0:
            R0 = Rv
        rows.append(
            {
                "m": m,
                "mL": m * L,
                "R": Rv,
                "R_over_R0": None if R0 is None else Rv / R0,
                "C_min": float(w.min()),
            }
        )
    rows[0]["R_over_R0"] = 1.0
    win = [r for r in rows if 0.0 < r["mL"] <= 8.0]
    c2 = bool(all(r["R_over_R0"] < THRESH for r in win))
    c1 = bool(0.0 < R0 < 1.0)
    ok = bool(c1 and c2)
    payload = {
        "task": "m9.12_A2_dirac_3d",
        "model": f"3d staggered fermion, grid {N}^3, region {L}^3",
        "pre_registered": {"C2_threshold": THRESH, "window": "0 < m L <= 8"},
        "R0": R0,
        "rows": rows,
        "C1": c1,
        "C2_PRIMARY": c2,
        "all_gates": ok,
        "verdict": "A2_DIRAC_3D_PASS" if ok else "A2_DIRAC_3D_FAIL",
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_12_A2_dirac_3d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
