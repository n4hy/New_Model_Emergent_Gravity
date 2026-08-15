#!/usr/bin/env python3
"""M9.24: deform the region at fixed H (Jacobson-style).

One staggered Hamiltonian, one vacuum C. Vary the region only.
S = Peschel of C restricted to the region. A_cut = bonds leaving it.

Shapes: Euclidean balls, cubes, taxicab (L1) diamonds, and one
ball shifted by one site. N=16. No hop perturbations.

PRE-REGISTERED:
  C0  Balls R=2,3,4,5: least-squares α in S = α A_cut + β is > 0.
  C1  Shifted ball R=4, center + (1,0,0): |S-S_centered|/S < 0.10
      (vacuum looks homogeneous; instrument).
  C2  PRIMARY. Fit (α,β) on balls only. RMS residual of CUBES
      exceeds RMS residual of the balls (shape matters beyond area).
  C3  Same as C2 for taxicab diamonds versus balls.
  C4  Closest ball/cube pair in A_cut: |S_ball-S_cube|/mean(S) > 0.02
      if A_cut agree within 15% (a finite shape piece exists).

Not claimed: eta=1/4G, Einstein, Planck, Bloch, de Sitter.

Writes ../data/m9_24_region_deform.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 16


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def staggered_H(n: int) -> np.ndarray:
    vol = n**3
    ham = np.zeros((vol, vol), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham[i, j] = ham[j, i] = -1.0
    return ham


def correlator(ham: np.ndarray) -> np.ndarray:
    ev, vecs = np.linalg.eigh(ham)
    return vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T


def peschel_s(cfull: np.ndarray, sites: list[int]) -> float:
    sl = np.array(sites, dtype=int)
    w = np.clip(np.linalg.eigvalsh(cfull[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def cut_area(coords: list[tuple[int, int, int]]) -> int:
    inside = set(coords)
    cuts = 0
    for x, y, z in coords:
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            if (x + d[0], y + d[1], z + d[2]) not in inside:
                cuts += 1
    return cuts


def ball_coords(n: int, radius: int, shift=(0, 0, 0)):
    c = n // 2
    cx, cy, cz = c + shift[0], c + shift[1], c + shift[2]
    out = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                    out.append((x, y, z))
    return out


def cube_coords(n: int, side: int):
    c = n // 2
    lo = c - side // 2
    out = []
    for x in range(lo, lo + side):
        for y in range(lo, lo + side):
            for z in range(lo, lo + side):
                if 0 <= x < n and 0 <= y < n and 0 <= z < n:
                    out.append((x, y, z))
    return out


def taxi_coords(n: int, t: int):
    c = n // 2
    out = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if abs(x - c) + abs(y - c) + abs(z - c) <= t:
                    out.append((x, y, z))
    return out


def measure(cfull, coords, name, kind, size):
    sites = [idx(*p) for p in coords]
    return {
        "name": name,
        "kind": kind,
        "size": size,
        "n": int(len(coords)),
        "A_cut": int(cut_area(coords)),
        "S": peschel_s(cfull, sites),
    }


def fit_ab(areas, ents):
    a = np.asarray(areas, dtype=float)
    s = np.asarray(ents, dtype=float)
    mat = np.column_stack([a, np.ones(len(a))])
    coef, _, _, _ = np.linalg.lstsq(mat, s, rcond=None)
    pred = mat @ coef
    rms = float(np.sqrt(np.mean((s - pred) ** 2)))
    return float(coef[0]), float(coef[1]), rms


def residuals(rows, alpha, beta):
    return np.array([r["S"] - alpha * r["A_cut"] - beta for r in rows])


def main() -> int:
    cfull = correlator(staggered_H(N))
    balls = [measure(cfull, ball_coords(N, r), f"ball_R{r}", "ball", r) for r in (2, 3, 4, 5)]
    cubes = [measure(cfull, cube_coords(N, s), f"cube_L{s}", "cube", s) for s in (3, 4, 5, 6)]
    taxis = [measure(cfull, taxi_coords(N, t), f"taxi_t{t}", "taxi", t) for t in (2, 3, 4, 5)]
    shifted = measure(cfull, ball_coords(N, 4, (1, 0, 0)), "ball_R4_shift", "ball_shift", 4)
    alpha, beta, rms_ball = fit_ab([b["A_cut"] for b in balls], [b["S"] for b in balls])
    res_cube = residuals(cubes, alpha, beta)
    res_taxi = residuals(taxis, alpha, beta)
    res_ball = residuals(balls, alpha, beta)
    rms_cube = float(np.sqrt(np.mean(res_cube**2)))
    rms_taxi = float(np.sqrt(np.mean(res_taxi**2)))
    # closest A_cut ball/cube pair
    best = None
    for b in balls:
        for c in cubes:
            rel_a = abs(b["A_cut"] - c["A_cut"]) / max(0.5 * (b["A_cut"] + c["A_cut"]), 1.0)
            if rel_a > 0.15:
                continue
            rel_s = abs(b["S"] - c["S"]) / max(0.5 * (b["S"] + c["S"]), 1e-12)
            cand = (rel_a, rel_s, b["name"], c["name"], b["A_cut"], c["A_cut"], b["S"], c["S"])
            if best is None or rel_a < best[0]:
                best = cand
    c0 = bool(alpha > 0.0)
    cen = next(b for b in balls if b["size"] == 4)
    c1 = bool(abs(shifted["S"] - cen["S"]) / cen["S"] < 0.10)
    c2 = bool(rms_cube > rms_ball)
    c3 = bool(rms_taxi > rms_ball)
    c4 = bool(best is not None and best[1] > 0.02)
    ok = bool(c0 and c1 and c2 and c3)
    payload = {
        "task": "m9.24_region_deform",
        "N": N,
        "alpha": alpha,
        "beta": beta,
        "rms_ball": rms_ball,
        "rms_cube": rms_cube,
        "rms_taxi": rms_taxi,
        "balls": balls,
        "cubes": cubes,
        "taxis": taxis,
        "shifted": shifted,
        "closest_pair": None
        if best is None
        else {
            "rel_A": best[0],
            "rel_S": best[1],
            "ball": best[2],
            "cube": best[3],
            "A_ball": best[4],
            "A_cube": best[5],
            "S_ball": best[6],
            "S_cube": best[7],
        },
        "C0_alpha_positive": c0,
        "C1_shift_homogeneous": c1,
        "C2_PRIMARY_cubes_worse": c2,
        "C3_taxi_worse": c3,
        "C4_matched_shape_gap": c4,
        "all_gates": ok,
        "verdict": "SHAPE_MATTERS" if ok else "AREA_ONLY_OR_FAIL",
        "not_claimed": ["eta=1/4G", "Einstein", "Planck", "Bloch", "de Sitter"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_24_region_deform.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
