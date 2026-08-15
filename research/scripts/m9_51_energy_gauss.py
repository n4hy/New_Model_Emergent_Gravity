#!/usr/bin/env python3
"""M9.51: Gauss from ∑δe. No κ. Slopes must not run with α.

Paper 60: κ(α)=2h(α)/(α ΔE) is not a coupling. The reusable
mass is P_flat=∑_B δe. This note rebuilds a(R)= −P_flat(R)/R²
and asks whether the star/sea shapes survive an α scan.

PRE-REGISTERED:
  STAR: open hop, N=12, src (6,6,6), σ=1.
  SEA:  periodic hop, band-edge transfer.
  α ∈ {0.01, 0.02, 0.04}.
  Balls R=2,3,4,5. a(R)= −P_flat(R)/R². No δS, no κ.
  Star slope on R with P_flat/∑δe > 0.95 (need ≥3).
  Sea slope on all four R.
  C_star  |slope+2| < 0.15 at every α
  C_sea   |slope−1| < |slope+2| at every α
  C_hold  PRIMARY. star slopes vary by < 0.10 peak-to-peak
          and sea slopes vary by < 0.10 peak-to-peak
          (shape is not an α artifact)

Not claimed: derived Einstein, 8πG, de Sitter, MODELS.md.

Writes ../data/m9_51_energy_gauss.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N = 12
SRC = (6, 6, 6)
SIGMA = 1.0
ALPHAS = (0.01, 0.02, 0.04)
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


def pflat_scan(de, ball_fn):
    return np.array([float(np.sum(de[ball_fn(SRC, r)])) for r in RADII])


def main() -> int:
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
    e0s = np.sum(ham_s * c0s, axis=1)
    dC_s = np.outer(right, right) - np.outer(left, left)

    ham_u = hop_pbc(N)
    ev_u, vecs_u = np.linalg.eigh(ham_u)
    il, ir = int(np.argmin(ev_u)), int(np.argmax(ev_u))
    occ_u = ev_u < 0.0
    c0u = vecs_u[:, occ_u] @ vecs_u[:, occ_u].T
    e0u = np.sum(ham_u * c0u, axis=1)
    dC_u = np.outer(vecs_u[:, ir], vecs_u[:, ir]) - np.outer(vecs_u[:, il], vecs_u[:, il])

    star_slopes, sea_slopes, rows = [], [], []
    c_star = True
    c_sea = True
    for alpha in ALPHAS:
        c1s = 0.5 * ((c0s + alpha * dC_s) + (c0s + alpha * dC_s).T)
        de_s = np.sum(ham_s * c1s, axis=1) - e0s
        p_s = pflat_scan(de_s, ball_open)
        m_s = float(np.sum(de_s))
        encl = p_s / m_s if m_s else p_s * 0.0
        fit_r = [r for r, e in zip(RADII, encl) if e > 0.95]
        a_s = -p_s / np.asarray(RADII, float) ** 2
        sl_s = slope_of(fit_r, [a_s[RADII.index(r)] for r in fit_r]) if len(fit_r) >= 3 else None
        if sl_s is None or abs(sl_s + 2.0) >= 0.15:
            c_star = False

        c1u = 0.5 * ((c0u + alpha * dC_u) + (c0u + alpha * dC_u).T)
        de_u = np.sum(ham_u * c1u, axis=1) - e0u
        p_u = pflat_scan(de_u, ball_pbc)
        a_u = -p_u / np.asarray(RADII, float) ** 2
        sl_u = slope_of(RADII, a_u)
        if abs(sl_u - 1.0) >= abs(sl_u + 2.0):
            c_sea = False

        star_slopes.append(sl_s)
        sea_slopes.append(sl_u)
        rows.append(
            {
                "alpha": alpha,
                "star_P": p_s.tolist(),
                "star_enclose": encl.tolist(),
                "star_fit_R": fit_r,
                "star_slope": sl_s,
                "sea_P": p_u.tolist(),
                "sea_slope": sl_u,
            }
        )

    ss = [s for s in star_slopes if s is not None]
    uu = list(sea_slopes)
    c_hold = bool(
        ss
        and uu
        and (max(ss) - min(ss)) < 0.10
        and (max(uu) - min(uu)) < 0.10
    )
    ok = bool(c_star and c_sea and c_hold)
    payload = {
        "task": "m9.51_energy_gauss",
        "rows": rows,
        "star_slopes": star_slopes,
        "sea_slopes": sea_slopes,
        "star_spread": (max(ss) - min(ss)) if ss else None,
        "sea_spread": (max(uu) - min(uu)) if uu else None,
        "C_star": c_star,
        "C_sea": c_sea,
        "C_hold_PRIMARY": c_hold,
        "all_gates": ok,
        "verdict": "ENERGY_GAUSS_HOLDS" if ok else "ENERGY_GAUSS_FAIL",
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_51_energy_gauss.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
