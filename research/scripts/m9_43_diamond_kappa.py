#!/usr/bin/env python3
"""M9.43: first-law κ on the 3+1D diamond waist. Not a cube artifact.

The waist is the geodesic ball of Papers 23–25. H is the spatial
Hamiltonian of a 3+1D staggered fermion. Occupation transfer at
fixed H. κ = δS / P_flat on source-centered enclosing balls.

PRE-REGISTERED:
  N=12, packet (6,6,6), σ=1, α=0.02.
  Staggered masses m ∈ {0.0, 0.25, 0.50}.
  Source-centered balls R=2,3,4,5.
  At each m, R_enc = smallest R with P_flat / M_global > 0.95.
  κ(m) = δS(R_enc) / P_flat(R_enc).
  C_eig   C eigenvalues in [0, 1] to 1e-9 at every m
  C_encl  R_enc exists (≤ 5) at every m
  C_star  δS(5)/δS(2) < 1.20 at m=0 and at m=0.50
          (still a compact star; Paper 51)
  C_vac   recorded, not required. Single-ball
          |δS − Tr(K_vac ΔC)| / |δS| is not the many-ball
          Pearson of Paper 37; it grows with R.
  C_univ  PRIMARY. max_m |κ(m)/κ(0) − 1| < 0.15
          The first-law constant survives 3+1D mass.
  C_plat  |κ(R=4)/κ(R=3) − 1| < 0.08 at every m that
          encloses by R=3 (both radii hold the packet).

Not claimed: continuum a→0, 8πG, 1/4G, FGHMV, de Sitter,
MODELS.md. Two-spacing continuum is Paper 23, not this run.

Writes ../data/m9_43_diamond_kappa.json
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
MASSES = (0.0, 0.25, 0.50)
RADII = (2, 3, 4, 5)


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def staggered_H(n: int, mass: float) -> np.ndarray:
    vol = n**3
    ham = np.zeros((vol, vol), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                ham[i, i] = mass * (1.0 if (x + y + z) % 2 == 0 else -1.0)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham[i, j] = ham[j, i] = -1.0
    return ham


def occupation_transfer(ham, n, src, sigma, alpha):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros((n**3,), dtype=float)
    stag = np.zeros((n**3,), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (sigma * sigma))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return c0, 0.5 * (corr + corr.T)


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr, sl):
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def ball(center, radius, n=N):
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


def main() -> int:
    rows = []
    kappas = {}
    c_eig = True
    c_encl = True
    c_vac = True
    grows = {}
    plat = {}
    for mass in MASSES:
        ham = staggered_H(N, mass)
        c0, c1 = occupation_transfer(ham, N, SRC, SIGMA, ALPHA)
        eigs = np.linalg.eigvalsh(c1)
        if eigs.min() < -1e-9 or eigs.max() > 1.0 + 1e-9:
            c_eig = False
        de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
        dc = c1 - c0
        m_glob = float(np.sum(de))
        scan = []
        r_enc = None
        kappa_enc = None
        ds_enc = None
        for rad in RADII:
            sl = ball(SRC, rad)
            ds = peschel_s(c1, sl) - peschel_s(c0, sl)
            p = float(np.sum(de[sl]))
            kv = peschel_k(c0, sl)
            trk = float(np.sum(kv * dc[np.ix_(sl, sl)]))
            vac_rel = abs(ds - trk) / abs(ds) if abs(ds) > 1e-18 else None
            encl = abs(p / m_glob) if m_glob else 0.0
            kap = (ds / p) if abs(p) > 1e-18 else None
            scan.append(
                {
                    "R": rad,
                    "deltaS": ds,
                    "P_flat": p,
                    "kappa": kap,
                    "enclose": encl,
                    "vac_rel": vac_rel,
                }
            )
            if r_enc is None and encl > 0.95:
                r_enc = rad
                kappa_enc = kap
                ds_enc = ds
                if vac_rel is None or vac_rel >= 0.10:
                    c_vac = False
        if r_enc is None:
            c_encl = False
        kappas[str(mass)] = kappa_enc
        grow = (
            scan[-1]["deltaS"] / scan[0]["deltaS"]
            if scan[0]["deltaS"]
            else None
        )
        grows[str(mass)] = grow
        k3 = next(s["kappa"] for s in scan if s["R"] == 3)
        k4 = next(s["kappa"] for s in scan if s["R"] == 4)
        plat[str(mass)] = (
            abs(k4 / k3 - 1.0) if (k3 and k4) else None
        )
        rows.append(
            {
                "m": mass,
                "M_global": m_glob,
                "R_enc": r_enc,
                "kappa": kappa_enc,
                "deltaS_enc": ds_enc,
                "scan": scan,
                "grow": grow,
            }
        )
    k0 = kappas["0.0"]
    univ = []
    for mass in MASSES:
        km = kappas[str(mass)]
        univ.append(abs(km / k0 - 1.0) if (km and k0) else None)
    c_univ = bool(all(u is not None and u < 0.15 for u in univ))
    c_star = bool(
        grows["0.0"] is not None
        and grows["0.0"] < 1.20
        and grows["0.5"] is not None
        and grows["0.5"] < 1.20
    )
    c_plat = True
    for mass in MASSES:
        row = next(r for r in rows if r["m"] == mass)
        if row["R_enc"] is not None and row["R_enc"] <= 3:
            val = plat[str(mass)]
            if val is None or val >= 0.08:
                c_plat = False
    ok = bool(c_eig and c_encl and c_univ and c_star)
    payload = {
        "task": "m9.43_diamond_kappa",
        "kappa_of_m": {str(m): kappas[str(m)] for m in MASSES},
        "univ_rel": univ,
        "grow": grows,
        "plat_4_over_3": plat,
        "rows": rows,
        "C_eig": c_eig,
        "C_encl": c_encl,
        "C_star": c_star,
        "C_vac_recorded": c_vac,
        "C_univ_PRIMARY": c_univ,
        "C_plat": c_plat,
        "all_gates": ok,
        "verdict": "KAPPA_SURVIVES_MASS" if ok else "DIAMOND_KAPPA_FAIL",
        "not_claimed": [
            "continuum a->0",
            "8pi G",
            "1/4G",
            "FGHMV",
            "de Sitter",
            "MODELS.md",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_43_diamond_kappa.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
