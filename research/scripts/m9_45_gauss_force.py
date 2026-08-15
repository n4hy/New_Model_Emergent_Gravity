#!/usr/bin/env python3
"""M9.45: first-law Gauss force. No Poisson solver.

Papers 48–54 imported DST Poisson to get a. Spherical Gauss
says 4π r² a_r = −4π G M(<r), so

    a(R) = − G M(R) / R² ,   M(R) = δS(R) / κ

with G=1 and lattice R. Two states, same rule:

  STAR   open hop, compact occupation transfer (Papers 47–50).
         M plateaus for R ≥ R_enc. Then a ~ 1/R².
  SEA    periodic band-edge transfer (Paper 54).
         M ~ V ~ R³. Then a ~ R.

PRE-REGISTERED:
  STAR: N=12, open hop, src (6,6,6), σ=1, α=0.02.
        Balls R=2,3,4,5. κ from smallest R with P/M_glob>0.95.
        M_FL(R)=δS(R)/κ. a(R)= −M_FL(R)/R².
        C_star PRIMARY. log-log slope of |a| vs R on
            {R: P/M_glob>0.95} has |α+2| < 0.15
            and at least three such R.
  SEA:  N=12, periodic hop, band edges, α=0.02.
        Same radii. κ from R=2 (density is flat).
        C_sea PRIMARY. slope of |a| vs R on R=2,3,4,5
            has |α−1| < 0.40 and |α−1| < |α+2|.
  C_split: star slope closer to −2 than to +1;
           sea slope closer to +1 than to −2.

Not claimed: derived Einstein, 8πG, FGHMV, de Sitter dual,
MODELS.md. Gauss is the integral Newtonian limit.

Writes ../data/m9_45_gauss_force.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
SRC = (6, 6, 6)
SIGMA = 1.0
ALPHA = 0.02
RADII = (2, 3, 4, 5)


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_open(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
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


def hop_pbc(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % n, (y + d[1]) % n, (z + d[2]) % n, n)
                    ham[i, j] = ham[j, i] = -1.0
    return ham


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def ball_open(center, radius, n=N):
    cx, cy, cz = center
    return np.array(
        [
            idx(x, y, z, n)
            for x in range(n)
            for y in range(n)
            for z in range(n)
            if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius
        ],
        dtype=int,
    )


def ball_pbc(center, radius, n=N):
    cx, cy, cz = center
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                dx = min((x - cx) % n, (cx - x) % n)
                dy = min((y - cy) % n, (cy - y) % n)
                dz = min((z - cz) % n, (cz - z) % n)
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    sl.append(idx(x, y, z, n))
    return np.array(sl, dtype=int)


def slope_of(radii, accs):
    lr = np.log(np.asarray(radii, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([lr, np.ones(len(radii))]), la, rcond=None
    )
    return float(coef[0])


def scan(c0, c1, de, radii, ball_fn):
    ds, pflat = [], []
    for rad in radii:
        sl = ball_fn(SRC, rad)
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pflat.append(float(np.sum(de[sl])))
    return np.asarray(ds), np.asarray(pflat)


def main() -> int:
    # --- STAR: open compact packet ---
    ham_s = hop_open(N)
    ev_s, vecs_s = np.linalg.eigh(ham_s)
    occ_s = ev_s < 0.0
    uo, uu = vecs_s[:, occ_s], vecs_s[:, ~occ_s]
    env = np.zeros(N**3)
    stag = np.zeros(N**3)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (SIGMA * SIGMA))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0s = uo @ uo.T
    c1s = 0.5 * (
        (c0s + ALPHA * (np.outer(right, right) - np.outer(left, left)))
        + (c0s + ALPHA * (np.outer(right, right) - np.outer(left, left))).T
    )
    de_s = np.sum(ham_s * c1s, axis=1) - np.sum(ham_s * c0s, axis=1)
    ds_s, p_s = scan(c0s, c1s, de_s, RADII, ball_open)
    m_glob_s = float(np.sum(de_s))
    encl_s = p_s / m_glob_s
    r_enc = next(r for r, e in zip(RADII, encl_s) if e > 0.95)
    kappa_s = float(ds_s[RADII.index(r_enc)] / p_s[RADII.index(r_enc)])
    m_fl_s = ds_s / kappa_s
    a_s = -m_fl_s / np.asarray(RADII, float) ** 2
    star_rs = [r for r, e in zip(RADII, encl_s) if e > 0.95]
    star_as = [a_s[RADII.index(r)] for r in star_rs]
    slope_star = slope_of(star_rs, star_as)
    c_star = bool(len(star_rs) >= 3 and abs(slope_star + 2.0) < 0.15)

    # --- SEA: periodic band edges ---
    ham_u = hop_pbc(N)
    ev_u, vecs_u = np.linalg.eigh(ham_u)
    i_l, i_r = int(np.argmin(ev_u)), int(np.argmax(ev_u))
    occ_u = ev_u < 0.0
    c0u = vecs_u[:, occ_u] @ vecs_u[:, occ_u].T
    left_u, right_u = vecs_u[:, i_l], vecs_u[:, i_r]
    c1u = 0.5 * (
        (c0u + ALPHA * (np.outer(right_u, right_u) - np.outer(left_u, left_u)))
        + (c0u + ALPHA * (np.outer(right_u, right_u) - np.outer(left_u, left_u))).T
    )
    de_u = np.sum(ham_u * c1u, axis=1) - np.sum(ham_u * c0u, axis=1)
    ds_u, p_u = scan(c0u, c1u, de_u, RADII, ball_pbc)
    kappa_u = float(ds_u[0] / p_u[0])
    m_fl_u = ds_u / kappa_u
    a_u = -m_fl_u / np.asarray(RADII, float) ** 2
    slope_sea = slope_of(RADII, a_u)
    c_sea = bool(abs(slope_sea - 1.0) < 0.40 and abs(slope_sea - 1.0) < abs(slope_sea + 2.0))
    c_split = bool(abs(slope_star + 2.0) < abs(slope_star - 1.0) and abs(slope_sea - 1.0) < abs(slope_sea + 2.0))
    ok = bool(c_star and c_sea and c_split)
    payload = {
        "task": "m9.45_gauss_force",
        "star": {
            "kappa": kappa_s,
            "R_enc": r_enc,
            "deltaS": ds_s.tolist(),
            "P_flat": p_s.tolist(),
            "enclose": encl_s.tolist(),
            "M_FL": m_fl_s.tolist(),
            "a": a_s.tolist(),
            "fit_R": star_rs,
            "slope": slope_star,
        },
        "sea": {
            "kappa": kappa_u,
            "deltaS": ds_u.tolist(),
            "P_flat": p_u.tolist(),
            "M_FL": m_fl_u.tolist(),
            "a": a_u.tolist(),
            "slope": slope_sea,
        },
        "C_star_PRIMARY": c_star,
        "C_sea_PRIMARY": c_sea,
        "C_split": c_split,
        "all_gates": ok,
        "verdict": "GAUSS_TWO_LAWS" if ok else "GAUSS_FAIL",
        "not_claimed": [
            "derived Einstein",
            "8pi G",
            "FGHMV",
            "de Sitter dual",
            "MODELS.md",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_45_gauss_force.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
