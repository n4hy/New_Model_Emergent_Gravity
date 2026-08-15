#!/usr/bin/env python3
"""M9.48: the first law subtracts the sea. E_vac does not gravitate.

If one fed ρ = e_i(C0) < 0 into Gauss, a would be outward.
The first law is δS = κ ∑_B δe, not κ ∑_B e. This run
measures that split on the same balls.

PRE-REGISTERED:
  Open hop, N=12, src (6,6,6), σ=1, α=0.02. No Poisson.
  Balls R=2,3,4,5.
  e_i(C) = ∑_j H_ij C_ij
  P_vac = ∑_B e(C0),  P_δe = ∑_B (e(C1)−e(C0)),  P_e = ∑_B e(C1)
  κ from smallest R with |P_δe|/∑|δe| > 0.95
  M_FL = δS/κ
  C_sign  P_vac < 0 and P_δe > 0 at every R
  C_scale |P_vac / P_δe| > 10 at R=5
          (raw energy is the sea, not the packet)
  C_fl    |M_FL / P_δe − 1| < 0.10 on enclosing R
  C_sub   PRIMARY. |ρ(δS, P_δe)| > 0.95
          and |ρ(δS, P_e)| < |ρ(δS, P_δe)|
          First law tracks excess energy, not raw e.
  Diagnostic a_vac(R) = −P_vac / R²  (outward if P_vac<0).
          Not a gate. Not a first-law force.

Not claimed: the sea is repulsive Newton, 8πG, de Sitter,
MODELS.md.

Writes ../data/m9_48_subtract.json
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


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


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
    ham = hop_open(N)
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
                env[i] = np.exp(-0.5 * rr / (SIGMA * SIGMA))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    c1 = 0.5 * (
        (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left)))
        + (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))).T
    )
    e0 = np.sum(ham * c0, axis=1)
    e1 = np.sum(ham * c1, axis=1)
    de = e1 - e0
    m_de = float(np.sum(de))
    ds, pvac, pde, pe = [], [], [], []
    for rad in RADII:
        sl = ball(SRC, rad)
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pvac.append(float(np.sum(e0[sl])))
        pde.append(float(np.sum(de[sl])))
        pe.append(float(np.sum(e1[sl])))
    ds, pvac, pde, pe = map(np.asarray, (ds, pvac, pde, pe))
    encl = np.abs(pde) / abs(m_de) if m_de else pde * 0.0
    k_idx = next((i for i, e in enumerate(encl) if e > 0.95), len(RADII) - 1)
    kappa = float(ds[k_idx] / pde[k_idx])
    m_fl = ds / kappa
    rho_de = pearson(ds, pde)
    rho_e = pearson(ds, pe)
    rho_vac = pearson(ds, pvac)
    a_vac = -pvac / np.asarray(RADII, float) ** 2
    c_sign = bool(all(pvac < 0.0) and all(pde > 0.0))
    c_scale = bool(abs(pvac[-1] / pde[-1]) > 10.0)
    enc_idx = [i for i, e in enumerate(encl) if e > 0.95]
    c_fl = bool(
        enc_idx
        and all(abs(m_fl[i] / pde[i] - 1.0) < 0.10 for i in enc_idx)
    )
    c_sub = bool(abs(rho_de) > 0.95 and abs(rho_e) < abs(rho_de))
    ok = bool(c_sign and c_scale and c_fl and c_sub)
    payload = {
        "task": "m9.48_subtract",
        "kappa": kappa,
        "deltaS": ds.tolist(),
        "P_vac": pvac.tolist(),
        "P_de": pde.tolist(),
        "P_e": pe.tolist(),
        "M_FL": m_fl.tolist(),
        "enclose": encl.tolist(),
        "rho_de": rho_de,
        "rho_e": rho_e,
        "rho_vac": rho_vac,
        "scale_R5": abs(float(pvac[-1] / pde[-1])),
        "a_vac_diagnostic": a_vac.tolist(),
        "a_vac_outward": bool(all(a > 0.0 for a in a_vac)),
        "C_sign": c_sign,
        "C_scale": c_scale,
        "C_fl": c_fl,
        "C_sub_PRIMARY": c_sub,
        "all_gates": ok,
        "verdict": "SEA_SUBTRACTED" if ok else "SUBTRACT_FAIL",
        "not_claimed": [
            "Fermi sea is repulsive Newton",
            "8pi G",
            "de Sitter",
            "MODELS.md",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_48_subtract.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
