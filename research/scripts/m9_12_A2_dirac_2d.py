#!/usr/bin/env python3
"""M9.12 A2: 2d staggered-mass free fermion (lattice Dirac).

Local ansatz: K_ij nonzero only for i=j or nearest neighbours on the
square lattice. A spacetime scalar is on-site. Range ≥ √2 cannot be δX.

PRE-REGISTERED:
  C1  R(0) in (0,1).
  C2  PRIMARY. For 0 < m L ≤ 8, R(m)/R(0) < 2.0.
  C3  Mutation: drop the diagonal; Roff(m_max)/Roff(0) > 1.2.
  C4  C_A spectrum in (0,1); K Hermitian.

Writes ../data/m9_12_A2_dirac_2d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N = 32
L = 12
THRESH = 2.0
EPS = 1e-12


def idx(x: int, y: int, n: int) -> int:
    return x * n + y


def staggered_H_2d(n: int, mass: float) -> np.ndarray:
    H = np.zeros((n * n, n * n), dtype=float)
    for x in range(n):
        for y in range(n):
            i = idx(x, y, n)
            H[i, i] = mass * (1.0 if (x + y) % 2 == 0 else -1.0)
            if x + 1 < n:
                j = idx(x + 1, y, n)
                H[i, j] = H[j, i] = -1.0
            if y + 1 < n:
                j = idx(x, y + 1, n)
                H[i, j] = H[j, i] = -1.0
    return H


def local_mask(n_reg: int) -> np.ndarray:
    """True if same site or grid nearest neighbour inside the L×L block."""
    mask = np.zeros((n_reg * n_reg, n_reg * n_reg), dtype=bool)
    for x in range(n_reg):
        for y in range(n_reg):
            i = idx(x, y, n_reg)
            mask[i, i] = True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < n_reg and 0 <= yy < n_reg:
                    mask[i, idx(xx, yy, n_reg)] = True
    return mask


def nn_mask(n_reg: int) -> np.ndarray:
    mask = np.zeros((n_reg * n_reg, n_reg * n_reg), dtype=bool)
    for x in range(n_reg):
        for y in range(n_reg):
            i = idx(x, y, n_reg)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < n_reg and 0 <= yy < n_reg:
                    mask[i, idx(xx, yy, n_reg)] = True
    return mask


def remainder_ratio(K: np.ndarray, mask: np.ndarray) -> float:
    loc = np.where(mask, K, 0.0)
    return float(np.linalg.norm(K - loc, "fro") / np.linalg.norm(K, "fro"))


def main() -> int:
    mid = N // 2
    sl = []
    for x in range(mid - L // 2, mid + L // 2):
        for y in range(mid - L // 2, mid + L // 2):
            sl.append(idx(x, y, N))
    sl = np.array(sl)
    loc = local_mask(L)
    nn = nn_mask(L)
    masses = [0.0, 0.1, 0.2, 0.4, 0.6]
    rows = []
    R0 = Roff0 = None
    for m in masses:
        H = staggered_H_2d(N, m)
        ev, vecs = np.linalg.eigh(H)
        C = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
        CA = C[np.ix_(sl, sl)]
        w, u = np.linalg.eigh(CA)
        w = np.clip(w, EPS, 1.0 - EPS)
        K = (u * np.log((1.0 - w) / w)) @ u.T
        K = 0.5 * (K + K.T)
        Rv = remainder_ratio(K, loc)
        Ro = remainder_ratio(K, nn)
        if m == 0.0:
            R0, Roff0 = Rv, Ro
        rows.append(
            {
                "m": m,
                "mL": m * L,
                "R": Rv,
                "R_over_R0": None if R0 is None else Rv / R0,
                "Roff": Ro,
                "Roff_over_0": None if Roff0 is None else Ro / Roff0,
                "C_min": float(w.min()),
                "C_max": float(w.max()),
                "K_herm": float(np.linalg.norm(K - K.T)),
            }
        )
    rows[0]["R_over_R0"] = 1.0
    rows[0]["Roff_over_0"] = 1.0
    win = [r for r in rows if 0.0 < r["mL"] <= 8.0]
    c1 = bool(0.0 < R0 < 1.0)
    c2 = bool(all(r["R_over_R0"] < THRESH for r in win))
    c3 = bool(rows[-1]["Roff_over_0"] > 1.2)
    c4 = bool(all(r["C_min"] > 0 and r["C_max"] < 1 and r["K_herm"] < 1e-9 for r in rows))
    ok = bool(c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.12_A2_dirac_2d",
        "model": f"2d staggered fermion, grid {N}x{N}, region {L}x{L}",
        "pre_registered": {"C2_threshold": THRESH, "window": "0 < m L <= 8"},
        "R0": R0,
        "rows": rows,
        "C1": c1,
        "C2_PRIMARY": c2,
        "C3_mutation": c3,
        "C4": c4,
        "all_gates": ok,
        "verdict": "A2_DIRAC_2D_PASS" if ok else "A2_DIRAC_2D_FAIL",
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_12_A2_dirac_2d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
