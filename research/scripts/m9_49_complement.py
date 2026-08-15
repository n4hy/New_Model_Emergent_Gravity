#!/usr/bin/env python3
"""M9.49: complement δS when mass is inside. Cosmological sign?

Paper 20 (metric SdS): T dS + dM = 0, so dS/dM < 0 on the
cosmological horizon. Ordinary Gauss balls have δS > 0
when enclosed energy rises (Papers 46–58).

This run asks whether the *complement* of an enclosing
ball has the cosmological minus sign.

PRE-REGISTERED:
  Open hop, N=12, src (6,6,6), σ=1, α=0.02.
  Balls R=3,4,5 (all enclose the packet; Paper 49).
  B = ball, B^c = complement.
  C_pure  |S0(B) − S0(B^c)| / S0(B) < 0.02
          (Fermi sea is pure: S(B)=S(B^c))
  C_in    δS(B) > 0 at every R          (ordinary first law)
  C_comp  PRIMARY. δS(B^c) < 0 at every R
          (cosmological-horizon sign)
  C_mix   S1(B) + S1(B^c) > S_global + 1e-9
          (perturbed state is mixed; no purity identity)
  S_global of C1 is 2 h(α), h(α)=−α log α−(1−α)log(1−α).

Not claimed: SdS, FGHMV, de Sitter dual, 8πG, MODELS.md.

Writes ../data/m9_49_complement.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
SRC = (6, 6, 6)
SIGMA = 1.0
ALPHA = 0.02
RADII = (3, 4, 5)


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_open(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham[i, j] = ham[j, i] = -1.0
    return ham


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def ball(center, radius, n=N):
    cx, cy, cz = center
    inside = []
    outside = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                    inside.append(i)
                else:
                    outside.append(i)
    return np.array(inside, dtype=int), np.array(outside, dtype=int)


def main() -> int:
    ham = hop_open(N)
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros(N**3)
    stag = np.zeros(N**3)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (SIGMA * SIGMA))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    c1 = 0.5 * (
        (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left)))
        + (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))).T
    )
    h_alpha = float(-ALPHA * np.log(ALPHA) - (1.0 - ALPHA) * np.log(1.0 - ALPHA))
    s_glob = 2.0 * h_alpha
    rows = []
    c_pure = True
    c_in = True
    c_comp = True
    c_mix = True
    for rad in RADII:
        sl_b, sl_c = ball(SRC, rad)
        s0b, s0c = peschel_s(c0, sl_b), peschel_s(c0, sl_c)
        s1b, s1c = peschel_s(c1, sl_b), peschel_s(c1, sl_c)
        dsb, dsc = s1b - s0b, s1c - s0c
        pure_rel = abs(s0b - s0c) / s0b if s0b else None
        if pure_rel is None or pure_rel >= 0.02:
            c_pure = False
        if dsb <= 0.0:
            c_in = False
        if dsc >= 0.0:
            c_comp = False
        if s1b + s1c <= s_glob + 1e-9:
            c_mix = False
        rows.append(
            {
                "R": rad,
                "n_B": int(len(sl_b)),
                "n_Bc": int(len(sl_c)),
                "S0_B": s0b,
                "S0_Bc": s0c,
                "S1_B": s1b,
                "S1_Bc": s1c,
                "dS_B": dsb,
                "dS_Bc": dsc,
                "pure_rel": pure_rel,
            }
        )
    ok_ordinary = bool(c_pure and c_in and c_mix)
    payload = {
        "task": "m9.49_complement",
        "S_global": s_glob,
        "rows": rows,
        "C_pure": c_pure,
        "C_in": c_in,
        "C_comp_PRIMARY": c_comp,
        "C_mix": c_mix,
        "all_gates": bool(ok_ordinary and c_comp),
        "verdict": (
            "COMPLEMENT_COSMO_SIGN"
            if (ok_ordinary and c_comp)
            else ("COMPLEMENT_PLUS" if (ok_ordinary and not c_comp) else "COMPLEMENT_FAIL")
        ),
        "not_claimed": ["SdS", "FGHMV", "de Sitter dual", "8pi G", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_49_complement.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    # exit 0 if the instrument held, even when C_comp fails (that's a result)
    return 0 if ok_ordinary else 1


if __name__ == "__main__":
    raise SystemExit(main())
