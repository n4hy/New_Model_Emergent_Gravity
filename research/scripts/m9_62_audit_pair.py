#!/usr/bin/env python3
"""M9.62 audit. N=11 exact basis, dps=80. Tries to REFUTE C_rho and C_src.

A=(2,5,5), B=(8,5,5), σ=0.9, α_A=0.015, α_B=0.045.
Enclosure: two-sided R=3 centres. Div: R=2 field.

Writes ../data/m9_62_audit_pair.json
"""

from __future__ import annotations

import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m9_60_exact as ex  # noqa: E402
import m9_62_pair_ent as pe  # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
mp.mp.dps = 80
N = 11
SRC_A = (2, 5, 5)
SRC_B = (8, 5, 5)
MID = (5, 5, 5)
SIGMA = mp.mpf("0.9")
ALPHA_A = mp.mpf("0.015")
ALPHA_B = mp.mpf("0.045")
R_ENC = 3
R_DIV = 2


def main() -> int:
    ex.N = N
    pe.N = N
    psi, eps = ex.open_1d(N)
    mask, n_occ = ex.occupy_mask(eps)
    occ_modes = pe.occupied_list(eps)
    la, ra = ex.packet_lr(psi, eps, mask, SRC_A, SIGMA)
    lb, rb = ex.packet_lr(psi, eps, mask, SRC_B, SIGMA)
    la, lb = ex.orthonormalize(la, lb)
    ra, rb = ex.orthonormalize(ra, rb)
    de = ex.site_de([(la, ra, ALPHA_A), (lb, rb, ALPHA_B)])
    m_tot = sum(de, mp.mpf(0))
    g4 = pe.gram4([la, lb, ra, rb])
    c4 = pe.c_in_basis(la, lb, ra, rb, ALPHA_A, ALPHA_B)
    gram_err = pe.max_gram_err(g4)
    off_c = pe.max_off_abs(c4)
    ev4, _ = mp.eigsy((c4 + c4.T) / 2)
    s4 = mp.mpf(0)
    for i in range(4):
        w = ev4[i]
        if w < pe.CLIP:
            w = pe.CLIP
        if w > 1 - pe.CLIP:
            w = 1 - pe.CLIP
        s4 -= w * mp.log(w) + (1 - w) * mp.log(1 - w)
    s_th = 2 * pe.h_bin(ALPHA_A) + 2 * pe.h_bin(ALPHA_B)
    sg_rel = abs(s4 - s_th) / s_th
    lo, hi = R_ENC, N - R_ENC
    centres = [
        (x, y, z)
        for x in range(lo, hi)
        for y in range(lo, hi)
        for z in range(lo, hi)
    ]
    cset = set(centres)
    keep = []
    for c in centres:
        okc = True
        for ax in range(3):
            cp, cm = list(c), list(c)
            cp[ax] += 1
            cm[ax] -= 1
            if tuple(cp) not in cset or tuple(cm) not in cset:
                okc = False
        if okc:
            keep.append(c)
    f_s, f_e, both_s, both_e = [], [], [], []
    for c in keep:
        sites, _ = pe.ball_sites(c, R_ENC, N)
        sl = [ex.idx(x, y, z, N) for (x, y, z) in sites]
        mass = sum((de[i] for i in sl), mp.mpf(0))
        c0b = pe.c0_block(psi, occ_modes, sites)
        dcb = pe.dC_block(la, ra, lb, rb, ALPHA_A, ALPHA_B, sites)
        ds = pe.peschel_s(c0b + dcb) - pe.peschel_s(c0b)
        fs, fe = ds / s4, mass / m_tot
        f_s.append(fs)
        f_e.append(fe)
        sa = (c[0] - SRC_A[0]) ** 2 + (c[1] - SRC_A[1]) ** 2 + (c[2] - SRC_A[2]) ** 2
        sb = (c[0] - SRC_B[0]) ** 2 + (c[1] - SRC_B[1]) ** 2 + (c[2] - SRC_B[2]) ** 2
        if sa <= R_ENC * R_ENC and sb <= R_ENC * R_ENC:
            both_s.append(abs(fs - 1))
            both_e.append(abs(fe - 1))
    rho = pe.pearson(f_s, f_e)
    rms = mp.sqrt(sum((f_s[i] - f_e[i]) ** 2 for i in range(len(f_s))) / len(f_s))
    med_s, med_e = ex.mp_median(both_s), ex.mp_median(both_e)
    map_m, map_a = ex.mass_map(de, R_DIV, N)
    gs = {}
    for c in map_m:
        gv = pe.field_g(c, map_m, map_a, m_tot)
        if gv is not None:
            gs[c] = gv

    def div_at(c):
        acc = mp.mpf(0)
        for ax in range(3):
            cp, cm = list(c), list(c)
            cp[ax] += 1
            cm[ax] -= 1
            tp, tm = tuple(cp), tuple(cm)
            if tp not in gs or tm not in gs:
                return None
            acc += (gs[tp][ax] - gs[tm][ax]) / 2
        return acc

    divs = {c: div_at(c) for c in gs}
    divs = {c: d for c, d in divs.items() if d is not None}
    d_mid = divs.get(MID)
    axis = [c for c in divs if c[1] == SRC_A[1] and c[2] == SRC_A[2]]
    c_a = min(axis, key=lambda c: abs(c[0] - SRC_A[0])) if axis else None
    c_b = min(axis, key=lambda c: abs(c[0] - SRC_B[0])) if axis else None
    d_a = divs.get(c_a) if c_a else None
    d_b = divs.get(c_b) if c_b else None
    src_max = None
    vals = [abs(v) for v in (d_a, d_b) if v is not None]
    if vals:
        src_max = max(vals)
    c_rho = bool(rho is not None and abs(rho) > mp.mpf("0.90"))
    c_rms = bool(rms < mp.mpf("0.15"))
    c_both = bool(
        both_s and med_s is not None and med_e is not None
        and med_s < mp.mpf("0.10")
        and med_e < mp.mpf("0.10")
    )
    c_src = bool(
        src_max is not None and d_mid is not None and src_max > 2 * abs(d_mid)
    )
    payload = {
        "task": "m9.62_audit_pair",
        "precision": {"eigh_hop": "none", "peschel": "mpmath eigsy dps=80", "n_occ": n_occ},
        "n_balls": len(f_s),
        "S_rel": ex.fnum(sg_rel),
        "gram_err": ex.fnum(gram_err),
        "C4_off": ex.fnum(off_c),
        "rho": ex.fnum(rho),
        "rms": ex.fnum(rms),
        "med_abs_fS_1": ex.fnum(med_s),
        "med_abs_fE_1": ex.fnum(med_e),
        "n_both": len(both_s),
        "div_near_A": ex.fnum(d_a),
        "div_near_B": ex.fnum(d_b),
        "div_mid": ex.fnum(d_mid),
        "C_rho": c_rho,
        "C_rms": c_rms,
        "C_both": c_both,
        "C_src": c_src,
        "verdicts": {
            "C_rho": "CONFIRMED" if c_rho else "REFUTED",
            "C_rms": "CONFIRMED" if c_rms else "REFUTED",
            "C_both": "CONFIRMED" if c_both else "REFUTED",
            "C_src": "CONFIRMED" if c_src else "REFUTED",
        },
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_62_audit_pair.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
