#!/usr/bin/env python3
"""Adversarial audit of M9.9: sympy implicit differentiation, no solver import."""

from __future__ import annotations

import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def main() -> int:
    M, r, ell = sp.symbols("M r ell", positive=True)
    f = 1 - 2 * M / r - r**2 / ell**2
    # F(M, r) = 0 ⇒ dr/dM = - F_M / F_r
    Fr = sp.diff(f, r)
    FM = sp.diff(f, M)
    drdM = sp.simplify(-FM / Fr)
    at0 = sp.simplify(drdM.subs({M: 0, r: ell}))
    kappa = sp.simplify(sp.Abs(Fr) / 2)
    kappa0 = kappa.subs({M: 0, r: ell})
    T = kappa / (2 * sp.pi)
    # T * dS with dS = 2π r dr, at M=0
    TdS_over_dM = sp.simplify((T * 2 * sp.pi * r * drdM).subs({M: 0, r: ell}))
    c1 = bool(at0 == -1)
    c2 = bool(sp.simplify(TdS_over_dM + 1) == 0)
    payload = {
        "task": "m9.9_audit_sds",
        "method": "sympy implicit differentiation of f(r_c(M))=0; no solver import",
        "dr_dM": str(drdM),
        "dr_dM_at_0": str(at0),
        "kappa_at_0": str(kappa0),
        "T_dS_over_dM_at_0": str(TdS_over_dM),
        "C1": c1,
        "C2": c2,
        "verdicts": {
            "C1": "CONFIRMED" if c1 else "REFUTED",
            "C2": "CONFIRMED" if c2 else "REFUTED",
            "FGHMV_in_dS": "NOT_CLAIMED",
            "value_of_Lambda": "NOT_CLAIMED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_9_audit_sds.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c1 and c2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
