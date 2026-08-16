#!/usr/bin/env python3
"""M9.60 audit. Exact open-hop basis, N=11, own pair.

Tries to REFUTE C_dir. mpmath dps=80. No LAPACK.

A=(2,5,5), B=(8,5,5), σ=0.9, α_A=0.015, α_B=0.045, R=3.

Writes ../data/m9_60_audit_exact.json
"""

from __future__ import annotations

import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m9_60_exact as ex  # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
mp.mp.dps = 80
N = 11
SRC_A = (2, 5, 5)
SRC_B = (8, 5, 5)
SIGMA = mp.mpf("0.9")
ALPHA_A = mp.mpf("0.015")
ALPHA_B = mp.mpf("0.045")
RADIUS = 3


def main() -> int:
    # temporarily point the helper module at N=11
    ex.N = N
    psi, eps = ex.open_1d(N)
    mask, n_occ = ex.occupy_mask(eps)
    la, ra = ex.packet_lr(psi, eps, mask, SRC_A, SIGMA)
    lb, rb = ex.packet_lr(psi, eps, mask, SRC_B, SIGMA)
    de_a = ex.site_de([(la, ra, ALPHA_A)])
    de_b = ex.site_de([(lb, rb, ALPHA_B)])
    la_p, lb_p = ex.orthonormalize(la, lb)
    ra_p, rb_p = ex.orthonormalize(ra, rb)
    de_ab = ex.site_de([(la_p, ra_p, ALPHA_A), (lb_p, rb_p, ALPHA_B)])
    m_a = sum(de_a, mp.mpf(0))
    m_b = sum(de_b, mp.mpf(0))
    m_tot = sum(de_ab, mp.mpf(0))
    add_rel = abs(m_tot - m_a - m_b) / abs(m_tot)
    pos_a = [mp.mpf(x) for x in SRC_A]
    pos_b = [mp.mpf(x) for x in SRC_B]
    masses, _ = ex.mass_map(de_ab, RADIUS, N)
    uneq = ex.score_pair(masses, m_a, m_b, m_tot, pos_a, pos_b)
    # score_pair hardcodes axis y=z=6; recompute axis for this lattice
    r_cm = [(m_a * pos_a[i] + m_b * pos_b[i]) / (m_a + m_b) for i in range(3)]
    null_x = ex.invsq_null_x(pos_a[0], pos_b[0], m_a, m_b)
    axis = [
        c
        for c in masses
        if c[1] == SRC_A[1]
        and c[2] == SRC_A[2]
        and ex.grad_at(c, masses) is not None
    ]
    c_null = min(axis, key=lambda c: abs(mp.mpf(c[0]) - null_x)) if axis else None
    c_cm = min(axis, key=lambda c: abs(mp.mpf(c[0]) - r_cm[0])) if axis else None
    g_null = (
        mp.sqrt(sum(x * x for x in ex.grad_at(c_null, masses))) if c_null else None
    )
    g_cm = (
        mp.sqrt(sum(x * x for x in ex.grad_at(c_cm, masses))) if c_cm else None
    )
    theta = uneq["_med_n"]
    med_cm = uneq["_med_cm"]
    c_dir = bool(uneq["n_keep"] >= 4 and theta is not None and theta < 20)
    c_notcm = bool(theta is not None and med_cm is not None and theta < med_cm)
    c_nullg = bool(g_null is not None and g_cm is not None and g_null < g_cm)
    payload = {
        "task": "m9.60_audit_exact",
        "precision": {
            "basis": "closed-form 1d open hop",
            "eigh": "none",
            "analysis": "mpmath dps=80",
            "n_occ": n_occ,
            "additivity_rel": ex.fnum(add_rel),
        },
        "M_A": ex.fnum(m_a),
        "M_B": ex.fnum(m_b),
        "M_AB": ex.fnum(m_tot),
        "n_keep": uneq["n_keep"],
        "median_ang_newton": uneq["median_ang_newton"],
        "median_ang_cm": uneq["median_ang_cm"],
        "null_x": ex.fnum(null_x),
        "cm_x": ex.fnum(r_cm[0]),
        "n_axis": len(axis),
        "grad_at_null": ex.fnum(g_null),
        "grad_at_cm": ex.fnum(g_cm),
        "C_dir_PRIMARY": c_dir,
        "C_notcm": c_notcm,
        "C_null": c_nullg,
        "verdicts": {
            "C_dir": "CONFIRMED" if c_dir else "REFUTED",
            "C_notcm": "CONFIRMED" if c_notcm else "REFUTED",
            "C_null": "CONFIRMED" if c_nullg else "REFUTED",
        },
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_60_audit_exact.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
