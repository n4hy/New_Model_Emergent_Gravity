#!/usr/bin/env python3
"""M9.14 A1: UV area-law coefficient on a 3+1D diamond waist.

S = -Tr[ C_A log C_A + (1-C_A) log(1-C_A) ] (Peschel).
Leading piece is an area law. α is the least-squares slope of
    S(R) = α A_cut(R) + β
at fixed lattice spacing, A_cut = number of bonds leaving the ball.

PRE-REGISTERED (locked before the run):
  C1      α(0) > 0.
  C2      PRIMARY. For every UV mass with m R_max <= 0.5,
            |α(m)-α(0)|/α(0) < 0.20.
  C3      Area-law fit beats volume-law fit at m=0:
            RMSE(S ≈ α A + β) < RMSE(S ≈ γ V + δ).
  C4      Fit uses at least three radii.

Not claimed: η = 1/4G, mean-zero curvature, quantum foam,
Einstein in vacuum, a value of Lambda, FGHMV in de Sitter.
Unsubtracted sea energy density is recorded as a diagnostic:
it is not zero, so this vacuum is not already mean-zero Einstein
curvature.

Writes ../data/m9_14_A1_diamond_4d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N = 16
RADII = (2, 3, 4, 5)
UV_MASSES = (0.0, 0.02, 0.05, 0.10)  # m R_max = 0, 0.10, 0.25, 0.50
IR_MASS = 0.40  # diagnostic only; area law need not die
EPS = 1e-12
REL_THRESH = 0.20


def idx(x: int, y: int, z: int, n: int) -> int:
    return (x * n + y) * n + z


def staggered_H_3d(n: int, mass: float) -> np.ndarray:
    vol = n**3
    ham = np.zeros((vol, vol), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                ham[i, i] = mass * (1.0 if (x + y + z) % 2 == 0 else -1.0)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham[i, j] = ham[j, i] = -1.0
    return ham


def ball_sites(n: int, radius: int) -> list[tuple[int, int, int]]:
    c = n // 2
    r2 = radius * radius
    coords: list[tuple[int, int, int]] = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= r2:
                    coords.append((x, y, z))
    return coords


def cut_area(coords: list[tuple[int, int, int]]) -> int:
    """Lattice area: bonds from a ball site to a non-ball neighbour."""
    inside = set(coords)
    cuts = 0
    for x, y, z in coords:
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            if (x + d[0], y + d[1], z + d[2]) not in inside:
                cuts += 1
    return cuts


def peschel_s(corr: np.ndarray, coords: list[tuple[int, int, int]], n: int) -> float:
    sl = np.array([idx(x, y, z, n) for x, y, z in coords], dtype=int)
    ev = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), EPS, 1.0 - EPS)
    return float(-np.sum(ev * np.log(ev) + (1.0 - ev) * np.log(1.0 - ev)))


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, RMSE of y = slope x + intercept."""
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    pred = mat @ coef
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return float(coef[0]), float(coef[1]), rmse


def row_for_mass(corr: np.ndarray, mass: float, balls: dict) -> dict:
    areas = []
    vols = []
    ents = []
    per_r = []
    for radius, coords in balls.items():
        area = cut_area(coords)
        ent = peschel_s(corr, coords, N)
        areas.append(area)
        vols.append(len(coords))
        ents.append(ent)
        per_r.append(
            {
                "R": radius,
                "A_cut": area,
                "V": len(coords),
                "S": ent,
            }
        )
    a = np.array(areas, dtype=float)
    v = np.array(vols, dtype=float)
    s = np.array(ents, dtype=float)
    alpha, beta, rmse_a = fit_slope(a, s)
    gamma, delta, rmse_v = fit_slope(v, s)
    return {
        "m": mass,
        "mR_max": mass * max(RADII),
        "alpha": alpha,
        "beta": beta,
        "rmse_area": rmse_a,
        "gamma_volume": gamma,
        "rmse_volume": rmse_v,
        "per_R": per_r,
    }


def main() -> int:
    balls = {r: ball_sites(N, r) for r in RADII}
    rows = []
    sea = []
    for mass in UV_MASSES:
        ham = staggered_H_3d(N, mass)
        ev, vecs = np.linalg.eigh(ham)
        occ = ev < 0.0
        corr = vecs[:, occ] @ vecs[:, occ].T
        sea.append(
            {
                "m": mass,
                "e_density": float(ev[occ].sum() / ev.size),
                "n_occ": int(occ.sum()),
            }
        )
        rows.append(row_for_mass(corr, mass, balls))

    ham_ir = staggered_H_3d(N, IR_MASS)
    ev_ir, vecs_ir = np.linalg.eigh(ham_ir)
    corr_ir = vecs_ir[:, ev_ir < 0.0] @ vecs_ir[:, ev_ir < 0.0].T
    ir_row = row_for_mass(corr_ir, IR_MASS, balls)

    a0 = rows[0]["alpha"]
    for r in rows:
        r["rel_to_0"] = abs(r["alpha"] - a0) / abs(a0) if a0 != 0.0 else None
    ir_row["rel_to_0"] = abs(ir_row["alpha"] - a0) / abs(a0) if a0 != 0.0 else None

    uv_win = [r for r in rows if r["mR_max"] <= 0.5]
    c1 = bool(a0 > 0.0)
    c2 = bool(all(r["rel_to_0"] < REL_THRESH for r in uv_win))
    c3 = bool(rows[0]["rmse_area"] < rows[0]["rmse_volume"])
    c4 = bool(len(RADII) >= 3)
    ok = bool(c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.14_A1_diamond_4d",
        "model": f"3+1D staggered fermion, grid {N}^3, balls R={list(RADII)}",
        "pre_registered": {
            "C2_rel_threshold": REL_THRESH,
            "UV_window": "m R_max <= 0.5",
            "area": "cut bonds leaving the ball",
            "radii": list(RADII),
        },
        "uv_rows": rows,
        "ir_row_diagnostic": ir_row,
        "sea_energy": sea,
        "alpha0": a0,
        "C1_alpha0_positive": c1,
        "C2_PRIMARY_UV_mass_independence": c2,
        "C3_area_beats_volume": c3,
        "C4_three_radii": c4,
        "all_gates": ok,
        "verdict": "A1_DIAMOND_4D_PASS" if ok else "A1_DIAMOND_4D_FAIL",
        "not_claimed": [
            "eta = 1/4G",
            "mean-zero curvature",
            "quantum foam",
            "Einstein vacuum",
            "value of Lambda",
            "FGHMV in de Sitter",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_14_A1_diamond_4d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
