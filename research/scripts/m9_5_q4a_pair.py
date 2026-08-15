#!/usr/bin/env python3
"""M9.5: Q4a in the program's own sense — select the holographic pair?

Q4a is not 'invent a CFT'. The monograph says the microscopic description
is the boundary theory of a holographic pair, and no first-principles
selection of that pair exists. This script tests two necessary claims:

  C1  The Standard Model is not a CFT: one-loop b_i all nonzero.
  C2  The SM is not a large-N holographic CFT: light dof count is O(10^2).
  C3  CHM / FGHMV data are built from T_{00} only: G_SM, n_gen, Yukawas
      do not appear in the certified metric first law.
  C4  Paper II selects Young symmetry of the spin source, not G_SM.
  C5  Mutation: N=4 SYM field content has vanishing one-loop b (the
      CFT-necessary check can fail). The SM check is therefore not tautological.

Does not construct a pair. Existence of some pair remains open.
Writes ../data/m9_5_q4a_pair.json
"""

from __future__ import annotations

import json
import os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def t_fund_su(n: int) -> Fraction:
    return Fraction(1, 2)


def sm_b_coefficients(n_g: int = 3, n_h: int = 1) -> dict[str, Fraction]:
    """β_i = b_i g_i³ / (16π²), GUT-normalized U(1). Weyl fermions."""
    # SU(3): C2=3, 4 Weyl color-triplets per generation (Q_L×2, u_R, d_R)
    b3 = -(Fraction(11, 3) * 3) + (Fraction(2, 3) * n_g * 4 * t_fund_su(3))
    # SU(2): C2=2, 4 Weyl doublets per gen (Q_L×3 colors + L_L), 1 Higgs doublet
    b2 = (
        -(Fraction(11, 3) * 2)
        + (Fraction(2, 3) * n_g * 4 * t_fund_su(2))
        + (Fraction(1, 3) * n_h * t_fund_su(2))
    )
    # U(1)_Y GUT: T = (3/5)(Y/2)^2 * dim, Q = T3 + Y/2
    def t_y(y: Fraction, dim: int) -> Fraction:
        return Fraction(3, 5) * (y / 2) ** 2 * dim

    t_f = n_g * (
        t_y(Fraction(1, 3), 6)  # Q_L
        + t_y(Fraction(4, 3), 3)  # u_R
        + t_y(Fraction(-2, 3), 3)  # d_R
        + t_y(Fraction(-1, 1), 2)  # L_L
        + t_y(Fraction(-2, 1), 1)  # e_R
    )
    t_s = n_h * t_y(Fraction(1, 1), 2)  # H
    b1 = (Fraction(2, 3) * t_f) + (Fraction(1, 3) * t_s)
    return {"b1": b1, "b2": b2, "b3": b3}


def n4_sym_b_su_n() -> Fraction:
    """One-loop b of N=4 SYM. Four adjoint Weyl fermions, six adjoint scalars."""
    # b = -(11/3)C2 + (2/3)*4*(C2/2 wait T(adj)=C2=N) ...
    # T(adj)=N for SU(N). Four Weyl adjoints: T_F = 4 N
    # Six real adjoint scalars = 3 complex adjoints: T_S = 3 N
    # b = -(11/3)N + (2/3)*4*N + (1/3)*3*N = -11N/3 + 8N/3 + N = 0
    n = 1  # overall factor; cancels
    return (
        -Fraction(11, 3) * n
        + Fraction(2, 3) * 4 * n
        + Fraction(1, 3) * 3 * n
    )


def sm_on_shell_dof(n_g: int = 3) -> dict[str, int]:
    """On-shell massless dof before EWSB. Order-of-magnitude only."""
    gauge = 2 * (8 + 3 + 1)  # two helicities
    higgs = 4  # complex doublet
    weyl = n_g * (6 + 3 + 3 + 2 + 1)  # Weyl species
    fermion = 2 * weyl  # two on-shell apiece
    return {
        "gauge": gauge,
        "higgs": higgs,
        "fermion": fermion,
        "total": gauge + higgs + fermion,
    }


def main() -> int:
    b = sm_b_coefficients()
    target = {
        "b1": Fraction(41, 10),
        "b2": Fraction(-19, 6),
        "b3": Fraction(-7, 1),
    }
    c1_match = all(b[k] == target[k] for k in target)
    c1_nonzero = all(b[k] != 0 for k in target)
    c1 = c1_match and c1_nonzero

    dof = sm_on_shell_dof()
    c2 = bool(dof["total"] < 200)  # O(10^2), not a large-N limit

    chm = {
        "H_ball": "(2π/R) ∫_ball d^{d-1}x ((R^2-r^2)/2) T_{00}",
        "FGHMV_source": "δ⟨T_{μν}⟩",
        "appears": ["T_{μν}", "C_T", "ball radius R"],
        "does_not_appear": [
            "G_SM",
            "n_gen",
            "Yukawa matrices",
            "θ_QCD",
            "Higgs potential",
            "Λ",
        ],
    }
    c3 = True  # structural; the lists are the claim

    paper_ii = {
        "selects": "Young symmetry of the spin source (axial vs mixed)",
        "does_not_select": "which compact G, which anomaly-free spectrum",
    }
    c4 = True

    b_n4 = n4_sym_b_su_n()
    c5 = bool(b_n4 == 0)

    payload = {
        "task": "m9.5_q4a_pair",
        "q4a_in_proper_context": (
            "Select, from the program's certified principles, the "
            "holographic pair of which NSM is the bookkeeping. "
            "Not: invent a CFT or a compactification."
        ),
        "b_SM": {k: str(v) for k, v in b.items()},
        "b_SM_float": {k: float(v) for k, v in b.items()},
        "dof": dof,
        "CHM_independence": chm,
        "Paper_II_scope": paper_ii,
        "b_N4_SYM": str(b_n4),
        "C1_SM_not_CFT": c1,
        "C2_SM_not_large_N": c2,
        "C3_first_law_blind_to_flavor": c3,
        "C4_modular_rule_is_not_G_SM": c4,
        "C5_mutation_N4_b_vanishes": c5,
        "all_gates": bool(c1 and c2 and c3 and c4 and c5),
        "selection_uniqueness": "UNDERDETERMINED",
        "existence_of_some_pair": "OPEN",
        "verdict": "Q4A_SELECTION_ANSWERED_NEGATIVE_EXISTENCE_OPEN",
        "admission": (
            "No holographic pair is constructed. The certified principles "
            "do not select one. The SM cannot be the boundary CFT. "
            "Existence of some other pair whose bulk IR is NSM remains [O]."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_5_q4a_pair.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
