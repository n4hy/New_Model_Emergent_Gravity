#!/usr/bin/env python3
"""M9.10 A2: is the modular Hamiltonian of a nonconformal free field local?

Jacobson 2016 conjectures, for nonconformal matter on a small diamond,
    δ⟨K⟩ = c δ⟨T_00⟩ + δX
with X a spacetime scalar (hence a *local* operator).

On a 1d lattice the local ansatz is a tridiagonal modular kernel:
diagonal = X + potential, first off-diagonal = kinetic T_00.
Any remainder with range ≥ 2 is nonlocal and cannot be δX.

Model: free tight-binding fermion with staggered mass (1d Dirac mass).
Vacuum Slater determinant. Interval reduced C. K = log((1-C)/C).

PRE-REGISTERED (locked before the numbers were read):
  C1  Massless baseline: R(0) = ||K-K_tri||_F/||K||_F is finite in (0,1).
  C2  PRIMARY. For every m with 0 < m L ≤ 8,
        R(m)/R(0) < 2.0.
      FAIL if any such m has R(m)/R(0) ≥ 2.
      (If A2 is true, extra X is diagonal and already in K_tri, so R
      should stay at the massless lattice artifact.)
  C3  Mutation: drop the diagonal from 'local'. Then R_off(m)/R_off(0)
      must exceed 1.2 at the largest m (a local mass term is detectable).
      If this mutation does not fire, the test cannot see a local X.
  C4  Spectrum of C_A sits in (ε, 1-ε); K is Hermitian.

A2 PASS only if C1-C4 all hold.
A2 FAIL (conjecture refuted for this field) if C2 fails.

Writes ../data/m9_10_A2_modular.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N_CHAIN = 240
L_INT = 32
THRESH = 2.0
EPS = 1e-12


def staggered_H(n: int, mass: float) -> np.ndarray:
    H = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        H[i, i + 1] = H[i + 1, i] = -1.0
    for i in range(n):
        H[i, i] = mass * (1.0 if i % 2 == 0 else -1.0)
    return H


def correlator_from_H(H: np.ndarray) -> np.ndarray:
    evals, vecs = np.linalg.eigh(H)
    filled = evals < 0.0
    v = vecs[:, filled]
    return v @ v.T


def modular_K(C: np.ndarray) -> np.ndarray:
    ev, u = np.linalg.eigh(C)
    ev = np.clip(ev, EPS, 1.0 - EPS)
    k_ev = np.log((1.0 - ev) / ev)
    return (u * k_ev) @ u.T


def tri_part(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    T = np.diag(np.diag(K))
    for i in range(n - 1):
        T[i, i + 1] = K[i, i + 1]
        T[i + 1, i] = K[i + 1, i]
    return T


def offdiag_part(K: np.ndarray) -> np.ndarray:
    """Nearest-neighbour only, no diagonal (mutation)."""
    n = K.shape[0]
    T = np.zeros_like(K)
    for i in range(n - 1):
        T[i, i + 1] = K[i, i + 1]
        T[i + 1, i] = K[i + 1, i]
    return T


def remainder_ratio(K: np.ndarray, local: np.ndarray) -> float:
    den = np.linalg.norm(K, "fro")
    return float(np.linalg.norm(K - local, "fro") / den)


def main() -> int:
    mid = N_CHAIN // 2
    sl = slice(mid - L_INT // 2, mid + L_INT // 2)
    masses = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    rows = []
    R0 = None
    Roff0 = None
    for m in masses:
        H = staggered_H(N_CHAIN, m)
        C = correlator_from_H(H)
        CA = C[sl, sl]
        ev = np.clip(np.linalg.eigvalsh(CA), EPS, 1.0 - EPS)
        K = modular_K(CA)
        herm = float(np.linalg.norm(K - K.T, "fro"))
        R = remainder_ratio(K, tri_part(K))
        Roff = remainder_ratio(K, offdiag_part(K))
        if m == 0.0:
            R0 = R
            Roff0 = Roff
        rows.append(
            {
                "m": m,
                "mL": m * L_INT,
                "R": R,
                "R_over_R0": None if R0 is None else R / R0,
                "Roff": Roff,
                "Roff_over_Roff0": None if Roff0 is None else Roff / Roff0,
                "C_min": float(ev.min()),
                "C_max": float(ev.max()),
                "K_herm": herm,
            }
        )

    assert R0 is not None and Roff0 is not None
    # fill m=0 ratio
    rows[0]["R_over_R0"] = 1.0
    rows[0]["Roff_over_Roff0"] = 1.0

    in_window = [r for r in rows if 0.0 < r["mL"] <= 8.0]
    ratios = [r["R_over_R0"] for r in in_window]
    c1 = bool(0.0 < R0 < 1.0)
    c2 = bool(all(x < THRESH for x in ratios)) if ratios else False
    c3 = bool(rows[-1]["Roff_over_Roff0"] > 1.2)
    c4 = bool(
        all(r["C_min"] > 0.0 and r["C_max"] < 1.0 and r["K_herm"] < 1e-10 for r in rows)
    )
    passed = bool(c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.10_A2_modular",
        "model": "1d staggered-mass tight-binding fermion, interval L=32, chain=240",
        "pre_registered": {
            "C2_threshold": THRESH,
            "window": "0 < m L <= 8",
            "local_ansatz": "tridiagonal (diag=X, nn=kinetic)",
        },
        "R0": R0,
        "rows": rows,
        "C1_massless_baseline": c1,
        "C2_PRIMARY_R_ratio_lt_2": c2,
        "C3_mutation_diagonal_detectable": c3,
        "C4_spectrum_ok": c4,
        "all_gates": passed,
        "verdict": "A2_PASS" if passed else "A2_FAIL_NONLOCAL",
        "admission": (
            "If C2 fails, Jacobson's local-X ansatz is false for this field: "
            "the modular kernel has range ≥ 2 that a spacetime scalar cannot absorb. "
            "1d lattice is a necessary-condition test, not a 4d diamond."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_10_A2_modular.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
