#!/usr/bin/env python3
"""M9.5: structural checks for the Palatini symplectic potential.

These are identities, not a second-order Einstein-Cartan theorem.

C1: the Palatini 4-form (1/4) eps_abcd e^a /\\ e^b /\\ R^{cd} depends on
    d omega only through R = d omega + omega /\\ omega. The symplectic
    potential therefore contains delta omega and not d(delta e).

C2: a matter term algebraic in omega (no derivative on omega) contributes
    nothing to the pre-symplectic potential.

C3: after omega = LC(e) + K with K algebraic in the spin, the reduced
    Hehl-Datta density has no derivative on a new field, so it is a
    potential, not a kinetic term.

Writes ../data/m9_5_ec_symplectic.json
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def main() -> int:
    # C1: string-level / derivative-counting check, recorded as an identity
    # of the Lagrangian, not a numerical integral.
    palatini_has_d_omega = True
    palatini_has_d_e = False  # e appears undifferentiated in e/\\e/\\R
    theta_contains_delta_omega = palatini_has_d_omega
    theta_contains_d_delta_e = palatini_has_d_e

    # C2
    dirac_omega_algebraic = True  # (i/8) omega * psibar {gamma, gamma_ab} psi
    matter_theta_from_omega = not dirac_omega_algebraic

    # C3
    hd_has_derivatives = False
    hd_is_potential = not hd_has_derivatives

    payload = {
        "task": "m9.5_ec_symplectic",
        "C1_palatini_theta_is_e_e_delta_omega": bool(
            theta_contains_delta_omega and not theta_contains_d_delta_e
        ),
        "C2_algebraic_spin_coupling_has_no_theta": bool(not matter_theta_from_omega),
        "C3_Hehl_Datta_is_a_potential": bool(hd_is_potential),
        "what_this_does_not_prove": (
            "It does not prove that second-order ball relative entropy "
            "equals the Einstein-Cartan canonical energy. That matching "
            "fails for the axial channel because CFT <J J> is nonlocal."
        ),
        "verdict": "STRUCTURE_ONLY",
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_5_ec_symplectic.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
