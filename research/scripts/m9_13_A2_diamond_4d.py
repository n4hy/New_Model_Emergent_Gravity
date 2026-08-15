#!/usr/bin/env python3
"""M9.13 A2: 3+1D free staggered fermion on a causal-diamond waist.

The physical theory is 4d spacetime (3-space + time). Virtual / field
modes of NSM live there, not on a 4-spatial-dimensional lattice. The
waist of a Minkowski causal diamond is a geodesic ball. Continuum
approach = two lattice spacings at matched dimensionless m L, with
L = 2 R the ball diameter.

Local ansatz: on-site + 6 spatial nearest neighbours (Peschel K).

PRE-REGISTERED (locked before the run):
  C1      0 < R(0) < 1 on each resolution.
  C2      PRIMARY. For 0 < m L <= 8, R(m)/R(0) < 2.0 on each resolution.
  C_CONT  C2 holds on both (N, R) = (12, 3) and (16, 5) at matched
          m L in {3, 6}. This is a two-spacing check, not a -> 0 proof.
  C4      C_A eigenvalues in (0, 1); K Hermitian.

Not claimed: Standard Model, eta = 1/4G, FGHMV in de Sitter, a value
of Lambda. Bisognano-Wichmann / CHM are cited continuum identities
for wedges / CFT balls; they are not outputs of this script.

Writes ../data/m9_13_A2_diamond_4d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

THRESH = 2.0
EPS = 1e-12
# (grid, radius). Diameter L = 2 R. Matched window masses give m L in {3, 6}.
RESOLUTIONS = (
    {"name": "coarse", "N": 12, "R": 3, "masses": [0.0, 0.5, 1.0]},
    {"name": "fine", "N": 16, "R": 5, "masses": [0.0, 0.3, 0.6]},
)


def idx(x: int, y: int, z: int, n: int) -> int:
    return (x * n + y) * n + z


def staggered_H_3d(n: int, mass: float) -> np.ndarray:
    """Spatial Hamiltonian of a 3+1D staggered fermion (hop = 1)."""
    vol = n**3
    h = np.zeros((vol, vol), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                h[i, i] = mass * (1.0 if (x + y + z) % 2 == 0 else -1.0)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        h[i, j] = h[j, i] = -1.0
    return h


def ball_sites(n: int, radius: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Integer points with (x-c)^2+(y-c)^2+(z-c)^2 <= R^2."""
    c = n // 2
    r2 = radius * radius
    sites: list[int] = []
    coords: list[tuple[int, int, int]] = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= r2:
                    sites.append(idx(x, y, z, n))
                    coords.append((x, y, z))
    return np.array(sites, dtype=int), coords


def local_mask(coords: list[tuple[int, int, int]]) -> np.ndarray:
    """True on the diagonal or a spatial nearest-neighbour pair inside the ball."""
    n_reg = len(coords)
    pos = {c: i for i, c in enumerate(coords)}
    mask = np.zeros((n_reg, n_reg), dtype=bool)
    for i, (x, y, z) in enumerate(coords):
        mask[i, i] = True
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nbr = (x + d[0], y + d[1], z + d[2])
            j = pos.get(nbr)
            if j is not None:
                mask[i, j] = True
    return mask


def remainder_ratio(k: np.ndarray, mask: np.ndarray) -> float:
    loc = np.where(mask, k, 0.0)
    return float(np.linalg.norm(k - loc, "fro") / np.linalg.norm(k, "fro"))


def run_resolution(n: int, radius: int, masses: list[float]) -> dict:
    sl, coords = ball_sites(n, radius)
    loc = local_mask(coords)
    diam = 2 * radius
    rows = []
    r0 = None
    for mass in masses:
        ham = staggered_H_3d(n, mass)
        ev, vecs = np.linalg.eigh(ham)
        corr = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
        ca = corr[np.ix_(sl, sl)]
        w, u = np.linalg.eigh(ca)
        w = np.clip(w, EPS, 1.0 - EPS)
        k = (u * np.log((1.0 - w) / w)) @ u.T
        k = 0.5 * (k + k.T)
        rv = remainder_ratio(k, loc)
        if mass == 0.0:
            r0 = rv
        rows.append(
            {
                "m": mass,
                "mL": mass * diam,
                "R": rv,
                "R_over_R0": None if r0 is None else rv / r0,
                "C_min": float(w.min()),
                "C_max": float(w.max()),
                "K_herm": float(np.linalg.norm(k - k.T)),
                "n_ball": int(sl.size),
            }
        )
    rows[0]["R_over_R0"] = 1.0
    win = [r for r in rows if 0.0 < r["mL"] <= 8.0]
    c1 = bool(r0 is not None and 0.0 < r0 < 1.0)
    c2 = bool(all(r["R_over_R0"] < THRESH for r in win))
    c4 = bool(
        all(r["C_min"] > 0.0 and r["C_max"] < 1.0 and r["K_herm"] < 1e-9 for r in rows)
    )
    return {
        "N": n,
        "R_ball": radius,
        "L_diam": diam,
        "n_ball": int(sl.size),
        "R0": r0,
        "rows": rows,
        "C1": c1,
        "C2_PRIMARY": c2,
        "C4": c4,
    }


def main() -> int:
    blocks = []
    for spec in RESOLUTIONS:
        block = run_resolution(spec["N"], spec["R"], spec["masses"])
        block["name"] = spec["name"]
        blocks.append(block)
    c_cont = bool(all(b["C2_PRIMARY"] for b in blocks))
    ok = bool(all(b["C1"] and b["C2_PRIMARY"] and b["C4"] for b in blocks) and c_cont)
    payload = {
        "task": "m9.13_A2_diamond_4d",
        "model": (
            "3+1D staggered fermion; diamond waist = geodesic ball; "
            "two spacings at matched m L"
        ),
        "pre_registered": {
            "C2_threshold": THRESH,
            "window": "0 < m L <= 8",
            "L": "ball diameter 2 R",
            "resolutions": ["N=12 R=3", "N=16 R=5"],
            "matched_mL": [3.0, 6.0],
        },
        "resolutions": blocks,
        "C_CONT": c_cont,
        "all_gates": ok,
        "verdict": "A2_DIAMOND_4D_PASS" if ok else "A2_DIAMOND_4D_FAIL",
        "not_claimed": [
            "continuum a -> 0 proof",
            "Standard Model",
            "eta = 1/4G",
            "FGHMV in de Sitter",
            "value of Lambda",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_13_A2_diamond_4d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
