#!/usr/bin/env python3
"""M9.12 A2: 1d free massive scalar (harmonic chain).

Local ansatz: tridiagonal modular kernels for (φ,π).
A spacetime scalar X is diagonal. Range ≥ 2 cannot be δX.

Vacuum correlators of the infinite chain, restricted to an interval.
Bosonic modular kernels from Casini-Huerta / Peschel:
  W = C^{1/2} P C^{1/2},   ξ = sqrt(eigen(W))
  β = log((ξ+1/2)/(ξ-1/2))
  K_φ, K_π reconstructed in the site basis; locality tested on both.

PRE-REGISTERED:
  C1  R_φ(0) and R_π(0) in (0,1).
  C2  PRIMARY. For 0 < m L ≤ 8, R_φ(m)/R_φ(0) < 2 and R_π(m)/R_π(0) < 2.
  C3  Mutation: drop diagonals; Roff_φ(m_max)/Roff_φ(0) > 1.2.
  C4  C,P positive definite; ξ > 1/2.

Writes ../data/m9_12_A2_scalar_1d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

L = 32
N_MOM = 4096
THRESH = 2.0
EPS = 1e-14


def vacuum_CP(n: int, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """C=⟨φφ⟩, P=⟨ππ⟩ on n consecutive sites of the infinite chain."""
    k = 2.0 * np.pi * np.arange(N_MOM) / N_MOM
    omega = np.sqrt(mass**2 + 4.0 * np.sin(k / 2.0) ** 2)
    omega = np.maximum(omega, 1e-10)
    dx = np.arange(n)[:, None] - np.arange(n)[None, :]
    phase = np.exp(1j * k[None, None, :] * dx[:, :, None])
    C = np.real(np.mean(phase / (2.0 * omega)[None, None, :], axis=2))
    P = np.real(np.mean(phase * (omega / 2.0)[None, None, :], axis=2))
    C = 0.5 * (C + C.T)
    P = 0.5 * (P + P.T)
    return C, P


def modular_kernels(C: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evals_c, u = np.linalg.eigh(C)
    evals_c = np.clip(evals_c, EPS, None)
    sqrtC = (u * np.sqrt(evals_c)) @ u.T
    inv_sqrtC = (u * (1.0 / np.sqrt(evals_c))) @ u.T
    W = sqrtC @ P @ sqrtC
    w, v = np.linalg.eigh(0.5 * (W + W.T))
    w = np.clip(w, EPS, None)
    xi = np.sqrt(w)
    xi = np.clip(xi, 0.5 + 1e-10, None)
    beta = np.log((xi + 0.5) / (xi - 0.5))
    # K_φ = inv_sqrtC v (β / (2ξ)) v^T inv_sqrtC
    # K_π = sqrtC v (β * 2ξ) v^T sqrtC
    mid = (v * (beta / (2.0 * xi))) @ v.T
    Kphi = inv_sqrtC @ mid @ inv_sqrtC
    midp = (v * (beta * 2.0 * xi)) @ v.T
    Kpi = sqrtC @ midp @ sqrtC
    Kphi = 0.5 * (Kphi + Kphi.T)
    Kpi = 0.5 * (Kpi + Kpi.T)
    return Kphi, Kpi, xi


def tri(K: np.ndarray) -> np.ndarray:
    T = np.diag(np.diag(K))
    n = K.shape[0]
    for i in range(n - 1):
        T[i, i + 1] = K[i, i + 1]
        T[i + 1, i] = K[i + 1, i]
    return T


def off_only(K: np.ndarray) -> np.ndarray:
    T = np.zeros_like(K)
    n = K.shape[0]
    for i in range(n - 1):
        T[i, i + 1] = K[i, i + 1]
        T[i + 1, i] = K[i + 1, i]
    return T


def R(K: np.ndarray, loc: np.ndarray) -> float:
    return float(np.linalg.norm(K - loc, "fro") / np.linalg.norm(K, "fro"))


def main() -> int:
    masses = [0.0, 0.05, 0.1, 0.2, 0.4]
    rows = []
    Rp0 = Rn0 = Ro0 = None
    for m in masses:
        C, P = vacuum_CP(L, m)
        Kphi, Kpi, xi = modular_kernels(C, P)
        rp = R(Kphi, tri(Kphi))
        rn = R(Kpi, tri(Kpi))
        ro = R(Kphi, off_only(Kphi))
        if m == 0.0:
            Rp0, Rn0, Ro0 = rp, rn, ro
        rows.append(
            {
                "m": m,
                "mL": m * L,
                "R_phi": rp,
                "R_pi": rn,
                "R_phi_over_0": None if Rp0 is None else rp / Rp0,
                "R_pi_over_0": None if Rn0 is None else rn / Rn0,
                "Roff_phi": ro,
                "Roff_over_0": None if Ro0 is None else ro / Ro0,
                "xi_min": float(xi.min()),
            }
        )
    rows[0]["R_phi_over_0"] = 1.0
    rows[0]["R_pi_over_0"] = 1.0
    rows[0]["Roff_over_0"] = 1.0
    win = [r for r in rows if 0.0 < r["mL"] <= 8.0]
    c1 = bool(0.0 < Rp0 < 1.0 and 0.0 < Rn0 < 1.0)
    c2 = bool(
        all(r["R_phi_over_0"] < THRESH and r["R_pi_over_0"] < THRESH for r in win)
    )
    c3 = bool(rows[-1]["Roff_over_0"] > 1.2)
    c4 = bool(all(r["xi_min"] > 0.5 for r in rows))
    ok = bool(c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.12_A2_scalar_1d",
        "model": "infinite harmonic chain, interval L=32",
        "pre_registered": {"C2_threshold": THRESH, "window": "0 < m L <= 8"},
        "R_phi_0": Rp0,
        "R_pi_0": Rn0,
        "rows": rows,
        "C1": c1,
        "C2_PRIMARY": c2,
        "C3_mutation": c3,
        "C4_xi": c4,
        "all_gates": ok,
        "verdict": "A2_SCALAR_1D_PASS" if ok else "A2_SCALAR_1D_FAIL",
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_12_A2_scalar_1d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
