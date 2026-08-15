#!/usr/bin/env python3
"""Adversarial audit of M9.7: prove the 1995 null lemma without the solver.

Mostly-minus. For k=(1,n), n·n=1:
  S_00 + 2 S_0i n^i + S_ij n^i n^j = 0 for all unit n
forces S_0i=0, S_ij=λ δ_ij, S_00=-λ, i.e. S=λ η.

Also recomputes SM b3 from the species table (no solver import).

Writes ../data/m9_7_audit_jacobson.json
"""

from __future__ import annotations

import json
import os
from fractions import Fraction

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def null_lemma_sympy() -> dict:
    S00, lmb = sp.symbols("S00 lambda", real=True)
    S0 = sp.Matrix(sp.symbols("S01 S02 S03", real=True))
    # Sij symmetric 3x3
    a, b, c, d, e, f = sp.symbols("a b c d e f", real=True)
    Sij = sp.Matrix([[a, d, e], [d, b, f], [e, f, c]])
    n1, n2, n3 = sp.symbols("n1 n2 n3", real=True)
    n = sp.Matrix([n1, n2, n3])
    q = S00 + 2 * (S0.dot(n)) + (n.T * Sij * n)[0]
    # Must vanish on the sphere. Use polarization:
    # n, -n kills the odd part ⇒ S0 = 0
    # n → basis e_i and (e_i+e_j)/√2
    conds = []
    # odd part
    for i in range(3):
        conds.append(S0[i])
    # even: q at e1, e2, e3 and at (e1+e2)/sqrt(2) etc after S0=0
    # q(e_i) = S00 + S_ii = 0
    # q((e1+e2)/√2) = S00 + (S11+S22)/2 + S12 = 0
    eqs = [
        S00 + a,
        S00 + b,
        S00 + c,
        S00 + (a + b) / 2 + d,
        S00 + (a + c) / 2 + e,
        S00 + (b + c) / 2 + f,
    ]
    vars_ = [S00, a, b, c, d, e, f]
    M, rhs = sp.linear_eq_to_matrix(eqs, vars_)
    # Six equations, seven unknowns: one-dimensional kernel.
    ker = M.nullspace()
    ok = len(ker) == 1
    v_out = None
    if ok:
        v = ker[0]
        # normalize to S00 = -1 ⇒ λ = 1, expect (a,b,c)=(1,1,1), off-diag 0
        v = sp.simplify(-v / v[0])
        v_out = [str(x) for x in v]
        ok = bool(v == sp.Matrix([-1, 1, 1, 1, 0, 0, 0]))
    return {
        "kernel_dim": len(ker),
        "kernel_S00_minus_1": v_out,
        "pass": bool(ok),
        "form": "S_00=-λ, S_ij=λ δ_ij, S_0i=0  (S=λ η, mostly-minus)",
    }


def sm_b3() -> dict:
    # 4 Weyl color-triplets per generation, 3 gens, T=1/2
    b3 = -11 + Fraction(2, 3) * 3 * 4 * Fraction(1, 2)
    return {"b3": str(b3), "pass": bool(b3 == -7)}


def main() -> int:
    lemma = null_lemma_sympy()
    b3 = sm_b3()
    payload = {
        "task": "m9.7_audit_jacobson",
        "method": "sympy polarization of S_kk=0 on the celestial sphere; no solver import",
        "null_lemma": lemma,
        "SM_b3": b3,
        "C1": lemma["pass"],
        "C5": b3["pass"],
        "verdicts": {
            "C1": "CONFIRMED" if lemma["pass"] else "REFUTED",
            "C5": "CONFIRMED" if b3["pass"] else "REFUTED",
            "Jacobson_as_Q2_substitute": "NOT_P",
            "MVEH": "NOT_CLAIMED",
            "2016_nonconformal": "NOT_CLAIMED",
        },
        "admission": (
            "The 1995 null lemma is an identity. The SM is not a CFT. "
            "Jacobson 2016's conformal half therefore does not apply to NSM matter."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_7_audit_jacobson.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (lemma["pass"] and b3["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
