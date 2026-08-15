#!/usr/bin/env python3
"""Audit of S^3 Wick identities. No solver import. Different method.

C2 by expanding 6/(I*ell)**2 in Python complex arithmetic.
C4 by a 2000-point trapezoid on chi, not sympy.

Writes ../data/m9_19_audit_s3.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def main() -> int:
    ell = 3.0
    r_wick = 6.0 / (1j * ell) ** 2
    c2 = bool(abs(r_wick.real + 6.0 / ell**2) < 1e-15 and abs(r_wick.imag) < 1e-15)

    chi = np.linspace(0.0, np.pi, 2001)
    w = np.sin(chi) ** 2
    mean = float(np.trapezoid(np.cos(chi) * w, chi) / np.trapezoid(w, chi))
    c4 = bool(abs(mean) < 1e-12)

    ok = bool(c2 and c4)
    payload = {
        "task": "m9.19_audit_s3",
        "method": "complex arithmetic + numpy trapezoid; no solver import",
        "R_wick_numeric": {"re": r_wick.real, "im": r_wick.imag},
        "haar_mean_cos_chi": mean,
        "C2": c2,
        "C4": c4,
        "verdicts": {"wick": "CONFIRMED" if ok else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_19_audit_s3.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
