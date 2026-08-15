#!/usr/bin/env python3
"""M9.19: S^3 with one imaginary axis / imaginary radius.

Textbook 3-geometry, not a virtual-particle theorem.

S^3 of radius rho, hyperspherical:
    ds^2 = rho^2 (d chi^2 + sin^2 chi d Omega_2^2)
Sectional curvature kappa = 1/rho^2.
Ricci scalar in 3d: R = 6 kappa = 6/rho^2.

PRE-REGISTERED:
  C1  R_sphere = 6/rho^2 for real rho > 0.
  C2  rho -> I * ell  gives R = -6/ell^2  (H^3 / Euclidean AdS).
  C3  Equator chi = pi/2 is a round S^2 of radius rho
      (induced metric rho^2 d Omega_2^2).
  C4  If one identifies k_axis = X_4 = cos chi, then the
      uniform (Haar) mean of k_axis on S^3 is identically 0.

C4 is an integral identity. It becomes 'mean-zero curvature'
only after the author identification k_axis = curvature.
That identification is not a gate.

Writes ../data/m9_19_s3_curvature_axis.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def ricci_scalar_s3(rho: sp.Symbol) -> sp.Expr:
    """Ricci scalar of the round S^3 metric, from Christoffel symbols."""
    chi, th, ph = sp.symbols("chi theta phi", positive=True)
    coords = (chi, th, ph)
    g = sp.diag(
        rho**2,
        rho**2 * sp.sin(chi) ** 2,
        rho**2 * sp.sin(chi) ** 2 * sp.sin(th) ** 2,
    )
    ginv = g.inv()
    n = 3
    gamma = [
        [[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)
    ]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                acc = 0
                for m in range(n):
                    acc += ginv[k, m] * (
                        sp.diff(g[m, j], coords[i])
                        + sp.diff(g[m, i], coords[j])
                        - sp.diff(g[i, j], coords[m])
                    )
                gamma[k][i][j] = sp.simplify(acc / 2)
    ric = sp.zeros(n)
    for i in range(n):
        for j in range(n):
            acc = 0
            for k in range(n):
                acc += sp.diff(gamma[k][i][j], coords[k]) - sp.diff(
                    gamma[k][i][k], coords[j]
                )
                for m in range(n):
                    acc += (
                        gamma[k][i][j] * gamma[m][k][m]
                        - gamma[m][i][k] * gamma[k][j][m]
                    )
            ric[i, j] = sp.simplify(acc)
    scalar = 0
    for i in range(n):
        for j in range(n):
            scalar += ginv[i, j] * ric[i, j]
    return sp.simplify(scalar)


def main() -> int:
    rho, ell, chi = sp.symbols("rho ell chi", positive=True)
    r_sphere = ricci_scalar_s3(rho)
    r_wick = sp.simplify(r_sphere.subs(rho, sp.I * ell))
    c1 = bool(r_sphere == 6 / rho**2)
    c2 = bool(r_wick == -6 / ell**2)

    # Induced metric on chi = pi/2: g_Omega = rho^2 * sin^2(chi) -> rho^2
    sin_eq = sp.simplify(sp.sin(chi).subs(chi, sp.pi / 2))
    c3 = bool(sin_eq == 1)

    # Haar measure on S^3: dmu ~ sin^2(chi) d chi d Omega, chi in [0, pi]
    # <cos chi> = int_0^pi cos(chi) sin^2(chi) d chi / int_0^pi sin^2(chi) d chi
    num = sp.integrate(sp.cos(chi) * sp.sin(chi) ** 2, (chi, 0, sp.pi))
    den = sp.integrate(sp.sin(chi) ** 2, (chi, 0, sp.pi))
    mean = sp.simplify(num / den)
    c4 = bool(mean == 0)

    # so(4) vs Lorentz / conformal (integer dimensions, not a dual)
    dim_so4 = 6
    dim_so31 = 6
    dim_so14 = 10
    dim_so24 = 15

    ok = bool(c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.19_s3_curvature_axis",
        "R_sphere": str(r_sphere),
        "R_wick": str(r_wick),
        "haar_mean_cos_chi": str(mean),
        "sin_equator": str(sin_eq),
        "isometry_dims": {
            "so4_S3": dim_so4,
            "so31_dS3_or_Lorentz": dim_so31,
            "so14_dS4": dim_so14,
            "so24_conf_or_AdS5": dim_so24,
        },
        "C1_sphere_Ricci": c1,
        "C2_wick_flips_sign": c2,
        "C3_equator_S2": c3,
        "C4_haar_mean_zero": c4,
        "all_gates": ok,
        "verdict": "S3_WICK_IDENTITIES_PASS" if ok else "S3_WICK_IDENTITIES_FAIL",
        "author_identification": (
            "X_4 = curvature is a guess, not a gate. "
            "Virtual particles on this S^3 is a guess, not a habitat."
        ),
        "not_claimed": [
            "virtual particles derived",
            "FGHMV in dS",
            "selection of Lambda",
            "so(1,4) contains CHM net",
            "mean-zero foam as a theorem of NSM",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_19_s3_curvature_axis.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
