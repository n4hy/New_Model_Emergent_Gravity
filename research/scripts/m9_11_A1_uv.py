#!/usr/bin/env python3
"""M9.11 A1: UV entanglement coefficient independent of IR mass.

For a 1d free fermion, S(L) ~ α log L + β in the UV (m L ≪ 1).
A1 asserts the leading UV piece is state/IR-mass independent.

PRE-REGISTERED:
  C1  α(0) ∈ (0.20, 0.45)  (c=1 interval: α = c/3 = 1/3).
  C2  PRIMARY. For every m with m L_big ≤ 0.5,
        |α(m)-α(0)|/|α(0)| < 0.20.
  C3  Mutation: at m L_big ≥ 4, |α-α(0)|/|α(0)| > 0.20
      (IR mass must be able to cut off the log).

Writes ../data/m9_11_A1_uv.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N_CHAIN = 240
EPS = 1e-12


def staggered_H(n: int, mass: float) -> np.ndarray:
    H = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        H[i, i + 1] = H[i + 1, i] = -1.0
    for i in range(n):
        H[i, i] = mass * (1.0 if i % 2 == 0 else -1.0)
    return H


def ee_interval(C: np.ndarray, L: int) -> float:
    mid = C.shape[0] // 2
    sl = slice(mid - L // 2, mid + L // 2)
    ev = np.clip(np.linalg.eigvalsh(C[sl, sl]), EPS, 1.0 - EPS)
    return float(-np.sum(ev * np.log(ev) + (1.0 - ev) * np.log(1.0 - ev)))


def alpha_of(mass: float, L_small: int = 16, L_big: int = 32) -> dict:
    H = staggered_H(N_CHAIN, mass)
    ev, vecs = np.linalg.eigh(H)
    C = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    s1 = ee_interval(C, L_small)
    s2 = ee_interval(C, L_big)
    alpha = (s2 - s1) / np.log(L_big / L_small)
    return {
        "m": mass,
        "mL_big": mass * L_big,
        "S_small": s1,
        "S_big": s2,
        "alpha": alpha,
    }


def main() -> int:
    uv_masses = [0.0, 0.005, 0.01]
    ir_mass = 0.2  # m L_big = 6.4
    uv_rows = [alpha_of(m) for m in uv_masses]
    ir_row = alpha_of(ir_mass)
    a0 = uv_rows[0]["alpha"]
    for r in uv_rows:
        r["rel_to_0"] = abs(r["alpha"] - a0) / abs(a0) if a0 != 0 else None
    ir_row["rel_to_0"] = abs(ir_row["alpha"] - a0) / abs(a0)

    c1 = bool(0.20 < a0 < 0.45)
    c2 = bool(all(r["rel_to_0"] < 0.20 for r in uv_rows))
    c3 = bool(ir_row["rel_to_0"] > 0.20)
    passed = bool(c1 and c2 and c3)
    payload = {
        "task": "m9.11_A1_uv",
        "uv_rows": uv_rows,
        "ir_row": ir_row,
        "alpha0": a0,
        "C1_alpha0_in_window": c1,
        "C2_PRIMARY_UV_mass_independence": c2,
        "C3_mutation_IR_cuts_off": c3,
        "all_gates": passed,
        "verdict": "A1_PASS" if passed else "A1_FAIL",
        "admission": (
            "1d free-fermion leading log coefficient. Not a 4d area law. "
            "η finite and IR-independent is tested only as this α."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_11_A1_uv.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
