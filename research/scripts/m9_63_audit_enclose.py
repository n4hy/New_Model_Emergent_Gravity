#!/usr/bin/env python3
"""M9.63 audit. N=11 exact. Tries to REFUTE C_fs.

A=(2,5,5), B=(8,5,5), σ=0.9, α_A=0.015, α_B=0.045.
ENC mid (5,5,5) R=5. ENC_OFF (5,5,4) R=5.
MISS on A, R=5. mpmath eigsy dps=80.

Writes ../data/m9_63_audit_enclose.json
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
SIGMA = mp.mpf("0.9")
ALPHA_A = mp.mpf("0.015")
ALPHA_B = mp.mpf("0.045")
BALLS = (
    ("ENC", (5, 5, 5), 5),
    ("ENC_OFF", (5, 5, 4), 5),
    ("MISS", (2, 5, 5), 5),
)


def site_in(src, center, radius):
    return sum((src[i] - center[i]) ** 2 for i in range(3)) <= radius * radius


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
    s_th = 2 * pe.h_bin(ALPHA_A) + 2 * pe.h_bin(ALPHA_B)
    c4 = pe.c_in_basis(la, lb, ra, rb, ALPHA_A, ALPHA_B)
    ev4, _ = mp.eigsy((c4 + c4.T) / 2)
    s4 = mp.mpf(0)
    for i in range(4):
        w = ev4[i]
        if w < pe.CLIP:
            w = pe.CLIP
        if w > 1 - pe.CLIP:
            w = 1 - pe.CLIP
        s4 -= w * mp.log(w) + (1 - w) * mp.log(1 - w)
    sg_rel = abs(s4 - s_th) / s_th
    rows = []
    by = {}
    for label, center, radius in BALLS:
        sites, _ = pe.ball_sites(center, radius, N)
        sl = [ex.idx(x, y, z, N) for (x, y, z) in sites]
        mass = sum((de[i] for i in sl), mp.mpf(0))
        c0b = pe.c0_block(psi, occ_modes, sites)
        dcb = pe.dC_block(la, ra, lb, rb, ALPHA_A, ALPHA_B, sites)
        ds = pe.peschel_s(c0b + dcb) - pe.peschel_s(c0b)
        fs, fe = ds / s4, mass / m_tot
        rec = {
            "label": label,
            "c": list(center),
            "R": radius,
            "n_sites": len(sites),
            "A_in": site_in(SRC_A, center, radius),
            "B_in": site_in(SRC_B, center, radius),
            "f_S": ex.fnum(fs),
            "f_E": ex.fnum(fe),
            "_fs": fs,
            "_fe": fe,
        }
        by[label] = rec
        rows.append({k: v for k, v in rec.items() if not k.startswith("_")})
        print(json.dumps(rows[-1]), flush=True)
    enc, off, miss = by["ENC"], by["ENC_OFF"], by["MISS"]
    c_fs = bool(abs(enc["_fs"] - 1) < mp.mpf("0.05"))
    c_fe = bool(abs(enc["_fe"] - 1) < mp.mpf("0.05"))
    c_off = bool(
        abs(off["_fs"] - 1) < mp.mpf("0.08") and abs(off["_fe"] - 1) < mp.mpf("0.08")
    )
    c_miss = bool(
        miss["_fe"] < mp.mpf("0.85")
        and abs(miss["_fs"] - miss["_fe"]) < mp.mpf("0.15")
    )
    payload = {
        "task": "m9.63_audit_enclose",
        "precision": {
            "eigh_hop": "none",
            "peschel": "mpmath eigsy dps=80",
            "n_occ": n_occ,
        },
        "S_rel": ex.fnum(sg_rel),
        "rows": rows,
        "C_fs_PRIMARY": c_fs,
        "C_fe": c_fe,
        "C_off": c_off,
        "C_miss": c_miss,
        "verdicts": {
            "C_fs": "CONFIRMED" if c_fs else "REFUTED",
            "C_fe": "CONFIRMED" if c_fe else "REFUTED",
            "C_off": "CONFIRMED" if c_off else "REFUTED",
            "C_miss": "CONFIRMED" if c_miss else "REFUTED",
        },
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_63_audit_enclose.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
