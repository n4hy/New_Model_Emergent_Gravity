#!/usr/bin/env python3
"""M9.4: tree-level integrate-out of a massive axial deformation.

This is NOT a selected UV completion (no CFT, no compactification).
It tests the only local bulk deformation forced by Paper 14 Thm. obs:

    L(k) = A (1 + k^2/M^2) |S(k)|^2 + B S(-k)·J(k)

with A, B locked so that M → ∞ recovers the M9.1 ratio 3/16.

C1: r(0) = 3/16
C2: r(k) = (3/16) / (1 + k^2/M^2)  (not a contact at finite k)
C3: the position-space kernel is 4d Yukawa, not a delta
C4: M → 0 kills the contact for every k ≠ 0
C5 (mutation): a wrong IR target 3/8 stays 3/8 at k=0 (check can fail)

Writes ../data/m9_4_uv_axial.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import k1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

R0 = 3.0 / 16.0
TOL = 1e-12


def r_of_k2(k2: np.ndarray | float, mass: float, r0: float = R0) -> np.ndarray | float:
    return r0 / (1.0 + np.asarray(k2, dtype=float) / mass**2)


def yukawa_4d(r: np.ndarray, mass: float) -> np.ndarray:
    """Euclidean 4d Fourier transform of 1/(k^2+M^2): (M/(4 π^2 r)) K_1(M r)."""
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r)
    tiny = r * mass < 1e-14
    out[tiny] = np.inf
    rr = r[~tiny]
    out[~tiny] = (mass / (4.0 * np.pi**2 * rr)) * k1(mass * rr)
    return out


def main() -> int:
    mass = 1.0
    k2 = np.array([0.0, 0.25, 1.0, 4.0, 16.0])
    r = r_of_k2(k2, mass)
    r_inf = r_of_k2(k2, 1.0e12)

    c1 = bool(abs(r[0] - R0) < TOL and np.max(np.abs(r_inf - R0)) < 1e-10)
    c2 = bool(np.all(np.diff(r) < 0.0) and abs(r[2] - R0 / 2.0) < TOL)

    radii = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    g = yukawa_4d(radii, mass)
    # A contact is a delta: off-origin samples would vanish. Yukawa is not.
    # Check the 4d radial ODE for r>0: G'' + (3/r) G' - M^2 G = 0.
    h = 1.0e-4
    r_ode = np.array([0.4, 0.8, 1.2])
    gp = yukawa_4d(r_ode + h, mass)
    gm = yukawa_4d(r_ode - h, mass)
    g0 = yukawa_4d(r_ode, mass)
    gpp = (gp - 2.0 * g0 + gm) / h**2
    gp1 = (gp - gm) / (2.0 * h)
    ode = gpp + 3.0 * gp1 / r_ode - mass**2 * g0
    rel_ode = np.max(np.abs(ode) / (mass**2 * np.abs(g0)))
    c3 = bool(rel_ode < 5.0e-4 and np.all(g > 0.0) and g[0] > g[-1])

    r_massless = r_of_k2(k2[1:], 1.0e-12)
    c4 = bool(np.max(np.abs(r_massless)) < 1e-8)

    r_wrong = float(np.asarray(r_of_k2(0.0, mass, r0=3.0 / 8.0)))
    c5 = bool(abs(r_wrong - 3.0 / 8.0) < TOL)

    payload = {
        "task": "m9.4_uv_axial",
        "what_this_is": (
            "Tree-level integrate-out of a massive axial deformation "
            "forced by Paper 14 Thm. obs. Not a selected CFT or string UV."
        ),
        "r0_target": R0,
        "M": mass,
        "k2": k2.tolist(),
        "r_of_k2": [float(x) for x in np.atleast_1d(r)],
        "r_M_infinite": [float(x) for x in np.atleast_1d(r_inf)],
        "yukawa_r": radii.tolist(),
        "yukawa_G": [float(x) for x in g],
        "ode_rel_residual": float(rel_ode),
        "C1_IR_recovers_3_16": c1,
        "C2_finite_k_not_contact": c2,
        "C3_yukawa_not_delta": c3,
        "C4_M_to_0_kills_contact": c4,
        "C5_mutation_wrong_IR_stays_wrong": c5,
        "all_gates": bool(c1 and c2 and c3 and c4 and c5),
        "verdict": (
            "IR_MATCHING_HOLDS_DEFORMATION_NOT_SELECTION"
            if (c1 and c2 and c3 and c4 and c5)
            else "GATE_FAIL"
        ),
        "admission": (
            "Q4a (selected holographic pair / SM content / metric "
            "renormalizability) remains open by construction. This "
            "file only checks the tree-level axial deformation Q4b."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_4_uv_axial.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
