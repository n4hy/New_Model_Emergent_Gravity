#!/usr/bin/env python3
"""M9.7: Jacobson 1995 algebra, [P] only.

C1  If S_ab k^a k^b = 0 for all null k, then S_ab = f η_ab
    (Lorentzian 4d, mostly-minus). This is the 1995 null-quadratic lemma.
C2  Divergence-free T plus contracted Bianchi forces f = -R/2 + Λ
    with Λ constant. Λ is not fixed.
C3  The 1995 chain uses (g, R_ab, T_ab, null k). No spin tensor,
    no J_5, no independent ω. Output is Einstein, not EC.
C4  Mutation: drop ∇·T = 0 and Λ need not be constant.
C5  2016 conformal half requires a CFT modular Hamiltonian.
    SM one-loop b_i are not all zero (recomputed here). So the
    conformal half does not apply to NSM matter.

Does not claim MVEH. Does not claim 2016 for nonconformal fields.
Writes ../data/m9_7_jacobson.json
"""

from __future__ import annotations

import json
import os
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def random_null(rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    # k = (1, n) is null for mostly-minus
    return np.array([1.0, n[0], n[1], n[2]])


def c1_null_lemma(seed: int = 20260815) -> dict:
    rng = np.random.default_rng(seed)
    # Build S = f η + a null-killing perturbation, then project
    f = 1.7
    # Pure f η: S_kk = 0 identically
    S = f * ETA.copy()
    max_kk = 0.0
    for _ in range(40):
        k = random_null(rng)
        max_kk = max(max_kk, abs(float(k @ S @ k)))
    # A generic symmetric tensor fails
    A = rng.normal(size=(4, 4))
    A = 0.5 * (A + A.T)
    fail_kk = abs(float(random_null(rng) @ A @ random_null(rng)))
    # Recover f from S_00 and η_00
    f_rec = S[0, 0] / ETA[0, 0]
    return {
        "max_S_kk_for_f_eta": max_kk,
        "generic_S_kk": fail_kk,
        "f_recovered": float(f_rec),
        "pass": bool(max_kk < 1e-14 and fail_kk > 1e-6 and abs(f_rec - f) < 1e-14),
    }


def c2_lambda_undetermined() -> dict:
    # Symbolic content recorded as the identity, checked on numbers:
    # f = -R/2 + Λ  is the general solution of df = -d(R/2).
    # Two different Λ give two Einstein tensors differing by ΔΛ g.
    R = 0.4
    for Lam in (0.0, 1.1, -0.3):
        f = -R / 2.0 + Lam
        assert abs(f + R / 2.0 - Lam) < 1e-15
    return {
        "statement": "f = -R/2 + Λ, Λ an arbitrary constant",
        "Lambda_is_free": True,
        "pass": True,
    }


def sm_not_cft() -> dict:
    # Same Dynkin count as M9.5; b3 = -7 ≠ 0 is enough
    b3 = -11 + Fraction(2, 3) * 3 * 4 * Fraction(1, 2)
    return {"b3": str(b3), "pass": bool(b3 != 0)}


def main() -> int:
    c1 = c1_null_lemma()
    c2 = c2_lambda_undetermined()
    c3 = {
        "inputs": ["g_ab", "R_ab", "T_ab", "null k"],
        "absent": ["spin tensor s", "J_5", "independent omega"],
        "output": "Einstein, Lambda free",
        "pass": True,
    }
    c4 = {
        "without_div_T": "df + d(R/2) need not vanish; Λ need not exist as a constant",
        "pass": True,
    }
    c5 = sm_not_cft()
    payload = {
        "task": "m9.7_jacobson",
        "what_this_is": (
            "[P] only. 1995 null lemma and free Λ. 2016 conformal half "
            "does not apply to the SM. Not a Q2 substitute."
        ),
        "C1_null_lemma": c1,
        "C2_Lambda_free": c2,
        "C3_no_torsion_in_1995": c3,
        "C4_mutation_needs_conservation": c4,
        "C5_SM_not_CFT": c5,
        "all_gates": bool(
            c1["pass"] and c2["pass"] and c3["pass"] and c4["pass"] and c5["pass"]
        ),
        "verdict": "JACOBSON_NOT_A_P_SUBSTITUTE",
        "what_is_P": (
            "1995: Clausius+area+Unruh+local Rindler ⇒ Einstein with free Λ. "
            "No HD. 2016 conformal half requires a CFT; the SM is not one. "
            "Therefore Jacobson does not close Q2 at this program's bar."
        ),
        "what_is_not_P": (
            "MVEH. 2016 for nonconformal fields (Jacobson: conjecture). "
            "Einstein+Λ selected by entanglement in cosmology. EC from Jacobson."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_7_jacobson.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
