#!/usr/bin/env python3
"""M9.63: pair enclosure on balls that actually hold both packets.

Paper 72: S_global = 2h(α_A)+2h(α_B). δS tracks M on
R=3, but those balls only touch the source sites
(C_both FAIL). Paper 50: R=4 holds 82% of ∑δe;
R=5 is the enclosing radius.

Three balls only (precision first, then efficiency):

  ENC      midpoint (6,6,6), R=5
  ENC_OFF  (5,6,6), R=5  (still holds both sites)
  MISS     centred on A, R=5  (B site outside)

Exact open-hop basis. Peschel = mpmath eigsy dps=80.
No hop LAPACK.

PRE-REGISTERED:
  N=12, A=(3,6,6), B=(9,6,6), σ=1.
  α_A=0.02, α_B=0.04.
  C_sg    |S_4 − 2h(α_A)−2h(α_B)| / S_4 < 1e-20
  C_fe    on ENC, |f_E − 1| < 0.05
  C_fs PRIMARY. on ENC, |f_S − 1| < 0.05
  C_off   on ENC_OFF, |f_S−1|<0.08 and |f_E−1|<0.08
  C_miss  on MISS, f_E < 0.85 and |f_S−f_E| < 0.15

Not claimed: derived Einstein, 8πG, de Sitter, MODELS.md.

Writes ../data/m9_63_enclose_pair.json
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
N = 12
SRC_A = (3, 6, 6)
SRC_B = (9, 6, 6)
SIGMA = mp.mpf(1)
ALPHA_A = mp.mpf("0.02")
ALPHA_B = mp.mpf("0.04")
BALLS = (
    ("ENC", (6, 6, 6), 5),
    ("ENC_OFF", (5, 6, 6), 5),
    ("MISS", (3, 6, 6), 5),
)


def site_in(src, center, radius):
    return sum((src[i] - center[i]) ** 2 for i in range(3)) <= radius * radius


def score_ball(label, center, radius, psi, occ_modes, la, ra, lb, rb, de, s4, m_tot):
    sites, _ = pe.ball_sites(center, radius, N)
    sl = [ex.idx(x, y, z, N) for (x, y, z) in sites]
    mass = sum((de[i] for i in sl), mp.mpf(0))
    c0b = pe.c0_block(psi, occ_modes, sites)
    dcb = pe.dC_block(la, ra, lb, rb, ALPHA_A, ALPHA_B, sites)
    s0 = pe.peschel_s(c0b)
    s1 = pe.peschel_s(c0b + dcb)
    ds = s1 - s0
    fs = ds / s4
    fe = mass / m_tot
    return {
        "label": label,
        "c": list(center),
        "R": radius,
        "n_sites": len(sites),
        "A_in": site_in(SRC_A, center, radius),
        "B_in": site_in(SRC_B, center, radius),
        "M": ex.fnum(mass),
        "dS": ex.fnum(ds),
        "f_S": ex.fnum(fs),
        "f_E": ex.fnum(fe),
        "_fs": fs,
        "_fe": fe,
    }


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
    s_th = 2 * pe.h_bin(ALPHA_A) + 2 * pe.h_bin(ALPHA_B)
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
        rec = score_ball(
            label, center, radius, psi, occ_modes, la, ra, lb, rb, de, s4, m_tot
        )
        by[label] = rec
        pub = {k: v for k, v in rec.items() if not k.startswith("_")}
        rows.append(pub)
        print(json.dumps(pub), flush=True)
    enc = by["ENC"]
    off = by["ENC_OFF"]
    miss = by["MISS"]
    c_sg = bool(sg_rel < mp.mpf("1e-20"))
    c_fe = bool(abs(enc["_fe"] - 1) < mp.mpf("0.05"))
    c_fs = bool(abs(enc["_fs"] - 1) < mp.mpf("0.05"))
    c_off = bool(
        abs(off["_fs"] - 1) < mp.mpf("0.08") and abs(off["_fe"] - 1) < mp.mpf("0.08")
    )
    c_miss = bool(
        miss["_fe"] < mp.mpf("0.85")
        and abs(miss["_fs"] - miss["_fe"]) < mp.mpf("0.15")
    )
    ok = bool(c_sg and c_fe and c_fs and c_off and c_miss)
    payload = {
        "task": "m9.63_enclose_pair",
        "precision": {
            "basis": "closed-form 1d open hop",
            "eigh_hop": "none",
            "peschel": "mpmath eigsy dps=80",
            "n_occ": n_occ,
        },
        "S_4": ex.fnum(s4),
        "S_theory": ex.fnum(s_th),
        "S_rel": ex.fnum(sg_rel),
        "gram_err": ex.fnum(pe.max_gram_err(g4)),
        "M_AB": ex.fnum(m_tot),
        "rows": rows,
        "C_sg": c_sg,
        "C_fe": c_fe,
        "C_fs_PRIMARY": c_fs,
        "C_off": c_off,
        "C_miss": c_miss,
        "all_gates": ok,
        "verdict": "PAIR_ENCLOSED" if ok else "PAIR_ENCLOSE_FAIL",
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_63_enclose_pair.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
