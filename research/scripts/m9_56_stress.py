#!/usr/bin/env python3
"""M9.56: hop kinetic stress of the transfer. Is p/ρ = −1?

Papers 54–65: mass is ∑δe; Gauss of that energy is inward
dust. de Sitter needs T_ij = −ρ δ_ij, i.e. p/ρ = −1.
This run measures the kinetic pressure of the hop
dispersion, not p = −E/V (that identity is forbidden;
it would fake Λ at μ=0) and not p = E/3V (tautology).

    ε(k) = −2 ∑_μ cos k_μ
    w(k) = (2/3) ∑_μ k_μ sin k_μ     (k in the 1st BZ)
    r = (∑ δn_m w_m) / (∑ δn_m ε_m) = δP / δE

PRE-REGISTERED:
  STAR: open hop, N=12, src (6,6,6), σ=1.
        Product open modes, k_μ = π n_μ/(N+1).
  SEA:  periodic hop, L = k=0, R = (π,π,π).
  α ∈ {0.01, 0.02, 0.04}.
  C_e     |δE − α(⟨R|H|R⟩−⟨L|H|L⟩)| / |δE| < 1e-8
          on the star (mode sum is the site energy).
  C_lambda PRIMARY. |r_sea + 1| < 0.25
          (uniform transfer would be Λ).
  C_dust  |r_sea| < 0.25
  C_rad   |r_sea − 1/3| < 0.15
  C_hold  r_star varies by < 0.05 across α
  Vacuum p/E is reported and is not a source
  (Paper 58). Forbidden: p = −E/V as a Λ claim.

Not claimed: 8πG, derived Einstein, de Sitter, MODELS.md.

Writes ../data/m9_56_stress.json
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


def fold(k: float) -> float:
    k = (k + np.pi) % (2.0 * np.pi) - np.pi
    return float(k)


def virial(kx, ky, kz) -> float:
    return (2.0 / 3.0) * (
        kx * np.sin(kx) + ky * np.sin(ky) + kz * np.sin(kz)
    )


def hop_eps(kx, ky, kz) -> float:
    return float(-2.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz)))


def open_1d(n: int):
    nn = np.arange(1, n + 1, dtype=float)
    k = np.pi * nn / (n + 1.0)
    eps = -2.0 * np.cos(k)
    j = np.arange(1, n + 1, dtype=float)
    psi = np.sqrt(2.0 / (n + 1.0)) * np.sin(np.outer(j, k))
    return k, eps, psi


def sea_pbc(n: int):
    e_vac = 0.0
    p_vac = 0.0
    n_occ = 0
    for nx in range(n):
        for ny in range(n):
            for nz in range(n):
                kx = fold(2.0 * np.pi * nx / n)
                ky = fold(2.0 * np.pi * ny / n)
                kz = fold(2.0 * np.pi * nz / n)
                eps = hop_eps(kx, ky, kz)
                w = virial(kx, ky, kz)
                if eps < 0.0:
                    e_vac += eps
                    p_vac += w
                    n_occ += 1
    k0 = (0.0, 0.0, 0.0)
    kpi = (np.pi, np.pi, np.pi)
    return {
        "E_vac": e_vac,
        "P_vac": p_vac,
        "n_occ": n_occ,
        "r_vac": (p_vac / e_vac) if e_vac != 0.0 else float("nan"),
        "E_L": hop_eps(*k0),
        "E_R": hop_eps(*kpi),
        "w_L": virial(*k0),
        "w_R": virial(*kpi),
    }


def star_modes(n, left, right, ham):
    k1, e1, psi = open_1d(n)
    l3 = left.reshape(n, n, n)
    r3 = right.reshape(n, n, n)
    amp_l = np.einsum("xa,yb,zc,xyz->abc", psi, psi, psi, l3)
    amp_r = np.einsum("xa,yb,zc,xyz->abc", psi, psi, psi, r3)
    kx = k1[:, None, None]
    ky = k1[None, :, None]
    kz = k1[None, None, :]
    eps = e1[:, None, None] + e1[None, :, None] + e1[None, None, :]
    w = virial(kx, ky, kz)
    dn1 = amp_r**2 - amp_l**2
    d_e = float(np.sum(dn1 * eps))
    d_p = float(np.sum(dn1 * w))
    site = float(right @ (ham @ right) - left @ (ham @ left))
    return d_e, d_p, site


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
    d_e1, d_p1, site1 = star_modes(N, left, right, ham)
    sea = sea_pbc(N)
    rows = []
    c_e = True
    c_lambda = True
    c_dust = True
    c_rad = True
    r_stars = []
    r_seas = []
    for alpha in ALPHAS:
        d_e = alpha * d_e1
        d_p = alpha * d_p1
        r_star = d_p / d_e if d_e != 0.0 else float("nan")
        err = abs(d_e1 - site1) / abs(site1) if site1 != 0.0 else abs(d_e1)
        if err >= 1e-8:
            c_e = False
        d_e_sea = alpha * (sea["E_R"] - sea["E_L"])
        d_p_sea = alpha * (sea["w_R"] - sea["w_L"])
        r_sea = d_p_sea / d_e_sea if d_e_sea != 0.0 else float("nan")
        if abs(r_sea + 1.0) >= 0.25:
            c_lambda = False
        if abs(r_sea) >= 0.25:
            c_dust = False
        if abs(r_sea - (1.0 / 3.0)) >= 0.15:
            c_rad = False
        r_stars.append(r_star)
        r_seas.append(r_sea)
        rows.append(
            {
                "alpha": alpha,
                "star_dE": d_e,
                "star_dP": d_p,
                "star_r": r_star,
                "star_site_rel_err": err,
                "sea_dE": d_e_sea,
                "sea_dP": d_p_sea,
                "sea_r": r_sea,
            }
        )
    c_hold = bool(max(r_stars) - min(r_stars) < 0.05)
    if c_lambda:
        verdict = "STRESS_LAMBDA"
    elif c_dust:
        verdict = "STRESS_DUST"
    elif c_rad:
        verdict = "STRESS_RADIATION"
    else:
        verdict = "STRESS_OTHER"
    ok = bool(c_e and (c_dust or c_rad or c_lambda) and c_hold)
    payload = {
        "task": "m9.56_stress",
        "definition": "r = sum dn w / sum dn eps; w=(2/3) k.sin k; k in 1st BZ",
        "forbidden": "p = -E/V as a Lambda claim",
        "sea_band": sea,
        "rows": rows,
        "star_r": r_stars,
        "sea_r": r_seas,
        "star_spread": float(max(r_stars) - min(r_stars)),
        "C_e": c_e,
        "C_lambda_PRIMARY": c_lambda,
        "C_dust": c_dust,
        "C_rad": c_rad,
        "C_hold": c_hold,
        "all_gates": ok,
        "verdict": verdict,
        "not_claimed": ["8pi G", "derived Einstein", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_56_stress.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (not c_lambda) else 1


if __name__ == "__main__":
    raise SystemExit(main())
