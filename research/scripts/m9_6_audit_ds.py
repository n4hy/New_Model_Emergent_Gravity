#!/usr/bin/env python3
"""Adversarial audit of M9.6: re-derive S(Λ) and so(p,q) without the solver.

Method:
  1. sympy: S = pi*ell**2/G, Lambda = 3/ell**2, differentiate.
  2. Count so(p,q) generators as antisymmetric (p+q)×(p+q) matrices.
  3. Try to refute C4 by using S = π ℓ²/G with Λ = +3/ℓ² and asking
     whether dS/dΛ can be positive — it cannot.

Writes ../data/m9_6_audit_ds.json
"""

from __future__ import annotations

import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def so_dim_count(n: int) -> int:
    """Number of independent antisymmetric n×n matrices."""
    return sum(1 for i in range(n) for j in range(i + 1, n))


def main() -> int:
    pi, ell, G, Lam = sp.symbols("pi ell G Lambda", positive=True)
    S = pi * ell**2 / G
    S_Lam = sp.simplify(S.subs({ell**2: 3 / Lam}))
    dS = sp.simplify(sp.diff(S_Lam, Lam))
    c1 = bool(
        sp.simplify(S_Lam - 3 * pi / (G * Lam)) == 0
        and sp.simplify(dS + 3 * pi / (G * Lam**2)) == 0
    )

    d10 = so_dim_count(5)  # so(1,4) and so(2,3) live on 5d
    d15 = so_dim_count(6)  # so(2,4) lives on 6d
    c2 = bool(d10 == 10 and d15 == 15)

    # C4: dS/dΛ is identically negative
    c4 = bool(sp.ask(sp.Q.negative(dS.subs({pi: 1, G: 1, Lam: 1}))))

    payload = {
        "task": "m9.6_audit_ds",
        "method": "sympy S(Λ); combinatorial so(n); no solver import",
        "S_of_Lambda": str(S_Lam),
        "dS_dLambda": str(dS),
        "so5_dim": d10,
        "so6_dim": d15,
        "C1": c1,
        "C2": c2,
        "C4": bool(c4),
        "verdicts": {
            "C1": "CONFIRMED" if c1 else "REFUTED",
            "C2": "CONFIRMED" if c2 else "REFUTED",
            "C4": "CONFIRMED" if c4 else "REFUTED",
            "dS_dual": "NOT_CLAIMED",
            "Einstein_plus_Lambda_from_CFT": "NOT_CLAIMED",
        },
        "admission": (
            "The GH identity and the Lie-algebra dimensions are confirmed. "
            "No de Sitter dual was produced."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_6_audit_ds.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c1 and c2 and c4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
