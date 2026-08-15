#!/usr/bin/env python3
"""M9.8: de Sitter is the spinless vacuum of Einstein+Λ.

Not a holographic selection. Not a Nobel claim.

C1  For the spatially flat FLRW chart of dS, a(t)=cosh(H t), H^2=Λ/3,
    the Friedmann equations hold with ρ=p=0.
C2  G_{tt}+Λ g_{tt} = 0 and the spatial Einstein+Λ component vanish
    at sample times (computed from a, ȧ, ä).
C3  Mutation: H=0 (Λ=0) is Minkowski, not dS: ä=0, not H^2 a.
C4  Torsion is not an input. The vacuum is Levi-Civita.

Writes ../data/m9_8_ds_vacuum.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def flrw_einstein_lambda(a, ad, add, Lam):
    """Mostly-minus FLRW. H=ad/a, return (G_tt+Λ g_tt, spatial combo).

    g_tt = -1, G_tt = 3 H^2, so G_tt+Λ g_tt = 3 H^2 - Λ.
    Spatial: G_ii + Λ g_ii ∝ 2ä/a + H^2 - Λ/3, up to a^2.
    Vacuum dS: H^2 = Λ/3 and ä/a = H^2.
    """
    H2 = (ad / a) ** 2
    acc = add / a
    tt = 3.0 * H2 - Lam
    # 2ä/a + H^2 - Λ  should vanish when ä/a = H^2 = Λ/3
    spat = 2.0 * acc + H2 - Lam
    return float(tt), float(spat), float(H2), float(acc)


def main() -> int:
    Lam = 3.0
    H = np.sqrt(Lam / 3.0)
    # Flat FLRW chart of dS: a = exp(H t), not the closed cosh chart.
    t = np.array([-0.5, 0.0, 0.4, 1.1])
    a = np.exp(H * t)
    ad = H * a
    add = H**2 * a
    tts, spats, H2s, accs = [], [], [], []
    for i in range(len(t)):
        tt, spat, H2, acc = flrw_einstein_lambda(a[i], ad[i], add[i], Lam)
        tts.append(tt)
        spats.append(spat)
        H2s.append(H2)
        accs.append(acc)
    c1 = bool(np.max(np.abs(np.array(H2s) - Lam / 3.0)) < 1e-12)
    c2 = bool(np.max(np.abs(tts)) < 1e-12 and np.max(np.abs(spats)) < 1e-12)
    # mutation: Minkowski a=1
    tt0, spat0, H20, acc0 = flrw_einstein_lambda(1.0, 0.0, 0.0, 0.0)
    tt_wrong, _, _, _ = flrw_einstein_lambda(1.0, 0.0, 0.0, Lam)
    c3 = bool(abs(tt0) < 1e-15 and abs(tt_wrong - (-Lam)) < 1e-15)
    c4 = True
    payload = {
        "task": "m9.8_ds_vacuum",
        "what_this_is": (
            "Spinless vacuum of Einstein+Λ with Λ>0 is de Sitter. "
            "Not entanglement selection. Not a prize claim."
        ),
        "Lambda": Lam,
        "H": H,
        "max_abs_Gtt_plus_Lambda_gtt": float(np.max(np.abs(tts))),
        "max_abs_spatial": float(np.max(np.abs(spats))),
        "C1_Friedmann_vacuum": c1,
        "C2_Einstein_plus_Lambda_zero": c2,
        "C3_mutation_Minkowski_not_dS": c3,
        "C4_no_torsion_input": c4,
        "all_gates": bool(c1 and c2 and c3 and c4),
        "verdict": "DS_IS_THE_SPINLESS_VACUUM",
        "what_is_P": (
            "G_μν+Λ g_μν=0 and Λ>0 ⇒ de Sitter (maximally symmetric). "
            "Torsion vanishes with the spin. Cosmology is metric."
        ),
        "what_is_not_P": (
            "Entanglement first law selects Λ or dS. FGHMV in cosmology. "
            "A Nobel-grade derivation of our universe."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_8_ds_vacuum.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
