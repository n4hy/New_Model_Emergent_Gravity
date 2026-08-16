#!/usr/bin/env python3
"""M9.61 audit. Exact basis N=11. Tries to REFUTE C_ang.

A=(2,5,5), B=(8,5,5), σ=0.9, α_A=0.015, α_B=0.045, R=3.
mpmath dps=80. No LAPACK.

Writes ../data/m9_61_audit_super.json
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
GRAD_FRAC = mp.mpf("1e-6")


def vnorm(v):
    return mp.sqrt(sum(x * x for x in v))


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vadd(a, b):
    return [a[i] + b[i] for i in range(3)]


def field_g(c, masses, areas, m_tot):
    gm = ex.grad_at(c, masses)
    if gm is None:
        return None
    gn = vnorm(gm)
    if gn <= GRAD_FRAC * abs(m_tot):
        return None
    mass = masses[c]
    area = mp.mpf(areas[c])
    if area == 0:
        return None
    scale = -mass / area / gn
    return [scale * gm[i] for i in range(3)]


def dist(c, src):
    return mp.sqrt(sum((mp.mpf(c[i]) - mp.mpf(src[i])) ** 2 for i in range(3)))


def main() -> int:
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
    m_ab = sum(de_ab, mp.mpf(0))
    map_a, area_a = ex.mass_map(de_a, RADIUS, N)
    map_b, area_b = ex.mass_map(de_b, RADIUS, N)
    map_ab, area_ab = ex.mass_map(de_ab, RADIUS, N)
    angs, rels, maps, far_angs = [], [], [], []
    for c in map_ab:
        ga = field_g(c, map_a, area_a, m_a)
        gb = field_g(c, map_b, area_b, m_b)
        gab = field_g(c, map_ab, area_ab, m_ab)
        if ga is None or gb is None or gab is None:
            continue
        gsum = vadd(ga, gb)
        ang = ex.angle_deg(gab, gsum)
        if ang is None:
            continue
        ng = vnorm(gab)
        rel = vnorm(vsub(gab, gsum)) / ng if ng != 0 else None
        ma, mb, mab = map_a[c], map_b[c], map_ab[c]
        map_rel = abs(mab - ma - mb) / (abs(ma) + abs(mb))
        angs.append(ang)
        if rel is not None:
            rels.append(rel)
        maps.append(map_rel)
        if dist(c, SRC_A) >= 3 and dist(c, SRC_B) >= 3:
            far_angs.append(ang)
    med_ang = ex.mp_median(angs)
    med_rel = ex.mp_median(rels)
    med_map = ex.mp_median(maps)
    med_far = ex.mp_median(far_angs)
    c_map = bool(maps and med_map < mp.mpf("0.05"))
    c_ang = bool(angs and med_ang < 20)
    c_rel = bool(rels and med_rel < mp.mpf("0.25"))
    c_far = bool(len(far_angs) >= 4 and med_far < 15)
    payload = {
        "task": "m9.61_audit_super",
        "precision": {
            "basis": "closed-form 1d open hop",
            "eigh": "none",
            "analysis": "mpmath dps=80",
            "n_occ": n_occ,
        },
        "n_keep": len(angs),
        "n_far": len(far_angs),
        "median_ang": ex.fnum(med_ang),
        "median_rel": ex.fnum(med_rel),
        "median_map_rel": ex.fnum(med_map),
        "median_far_ang": ex.fnum(med_far),
        "C_map": c_map,
        "C_ang_PRIMARY": c_ang,
        "C_rel": c_rel,
        "C_far": c_far,
        "verdicts": {
            "C_map": "CONFIRMED" if c_map else "REFUTED",
            "C_ang": "CONFIRMED" if c_ang else "REFUTED",
            "C_rel": "CONFIRMED" if c_rel else "REFUTED",
            "C_far": "CONFIRMED" if c_far else "REFUTED",
        },
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_61_audit_super.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
