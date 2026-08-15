#!/usr/bin/env python3
"""M9.46: Gauss slope vs packet width. Attractive dust, not Λ.

Paper 54/55 called the uniform sea a Newtonian Λ signature.
Interior Newton of positive uniform ρ is a = −(4πGρ/3) r
(inward). de Sitter / Λ>0 is a = +(Λ/3) r (outward).
This run records the sign and interpolates the slope.

PRE-REGISTERED:
  Open hop, N=12, src (6,6,6), α=0.02.
  σ ∈ {1.0, 2.0, 4.0, 8.0}.
  First-law Gauss a(R)= −(δS/κ)/R², κ from smallest R
  with P/M_glob>0.95, else from R=5.
  Slope of |a| vs R on R=2,3,4,5 (all four; growth is
  the point of the scan).
  PBC band-edge sea (Paper 54) as the flat endpoint.
  C_star  σ=1: |slope+2| < 0.20
  C_mono  slope(σ) is non-decreasing in σ
  C_wide  σ=8 slope > slope(σ=1) + 0.30
  C_sea   PBC sea slope closer to +1 than to −2
  C_in    PRIMARY. a(R)<0 at every R, every σ, and the sea.
          The force is inward. Not de Sitter.

Not claimed: derived Einstein, 8πG, FGHMV, de Sitter,
MODELS.md. Paper 54's "Newtonian Λ" is hereby corrected.

Writes ../data/m9_46_sigma.json
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
ALPHA = 0.02
SIGMAS = (1.0, 2.0, 4.0, 8.0)
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


def gauss_scan(c0, c1, de, ball_fn):
    ds, pflat = [], []
    for rad in RADII:
        sl = ball_fn(SRC, rad)
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pflat.append(float(np.sum(de[sl])))
    ds, pflat = np.asarray(ds), np.asarray(pflat)
    m_glob = float(np.sum(de))
    encl = pflat / m_glob if m_glob else pflat * 0.0
    if np.any(encl > 0.95):
        k_idx = next(i for i, e in enumerate(encl) if e > 0.95)
    else:
        k_idx = len(RADII) - 1
    kappa = float(ds[k_idx] / pflat[k_idx]) if abs(pflat[k_idx]) > 1e-18 else None
    m_fl = ds / kappa if kappa else ds * np.nan
    accs = -m_fl / np.asarray(RADII, float) ** 2
    return {
        "deltaS": ds.tolist(),
        "P_flat": pflat.tolist(),
        "enclose": encl.tolist(),
        "kappa": kappa,
        "M_FL": m_fl.tolist(),
        "a": accs.tolist(),
        "slope": slope_of(RADII, accs),
        "inward": bool(all(a < 0.0 for a in accs)),
    }


def packet(ham, sigma, ev=None, vecs=None):
    if ev is None or vecs is None:
        ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros(N**3)
    stag = np.zeros(N**3)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (sigma * sigma))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    c1 = c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))
    c1 = 0.5 * (c1 + c1.T)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    return c0, c1, de


def main() -> int:
    ham = hop_open(N)
    ev_o, vecs_o = np.linalg.eigh(ham)
    rows = []
    slopes = []
    inward = True
    for sig in SIGMAS:
        c0, c1, de = packet(ham, sig, ev_o, vecs_o)
        row = gauss_scan(c0, c1, de, ball_open)
        row["sigma"] = sig
        rows.append(row)
        slopes.append(row["slope"])
        if not row["inward"]:
            inward = False

    hamu = hop_pbc(N)
    evu, vecu = np.linalg.eigh(hamu)
    il, ir = int(np.argmin(evu)), int(np.argmax(evu))
    occu = evu < 0.0
    c0u = vecu[:, occu] @ vecu[:, occu].T
    c1u = c0u + ALPHA * (
        np.outer(vecu[:, ir], vecu[:, ir]) - np.outer(vecu[:, il], vecu[:, il])
    )
    c1u = 0.5 * (c1u + c1u.T)
    deu = np.sum(hamu * c1u, axis=1) - np.sum(hamu * c0u, axis=1)
    sea = gauss_scan(c0u, c1u, deu, ball_pbc)
    sea["sigma"] = "pbc_edge"
    if not sea["inward"]:
        inward = False

    c_star = bool(abs(slopes[0] + 2.0) < 0.20)
    c_mono = bool(all(slopes[i] <= slopes[i + 1] + 1e-12 for i in range(len(slopes) - 1)))
    c_wide = bool(slopes[-1] > slopes[0] + 0.30)
    c_sea = bool(abs(sea["slope"] - 1.0) < abs(sea["slope"] + 2.0))
    c_in = bool(inward)
    ok = bool(c_star and c_mono and c_wide and c_sea and c_in)
    payload = {
        "task": "m9.46_sigma",
        "scan": rows,
        "slopes": slopes,
        "sea": sea,
        "C_star": c_star,
        "C_mono": c_mono,
        "C_wide": c_wide,
        "C_sea": c_sea,
        "C_in_PRIMARY": c_in,
        "all_gates": ok,
        "verdict": "DUST_NOT_LAMBDA" if ok else "SIGMA_FAIL",
        "correction": "Paper 54 Newtonian Lambda is withdrawn. Inward a is dust.",
        "not_claimed": ["de Sitter", "8pi G", "derived Einstein", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_46_sigma.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
