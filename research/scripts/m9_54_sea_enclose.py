#!/usr/bin/env python3
"""M9.54: is enclosure of 2h(α) a sea fact, or only a packet fact?

Paper 63: on a compact packet, f_S = δS/S_global tracks
f_E = P_flat/M_global. Nested balls cannot decide the sea:
A and V are collinear (Paper 42). Paper 57 vacuum S is
area-law; Paper 44 sea-transfer P_flat is volume-law.

This run uses the PBC band-edge transfer (the sea) and
slabs, whose area is independent of thickness.

PRE-REGISTERED:
  N=12, periodic hop. L = min-ev (k=0), R = max-ev (π,π,π).
  α ∈ {0.01, 0.02, 0.04}. S_global = 2h(α). M_global = ∑δe.
  Regions (one centre; the density is flat):
    balls R=2,3,4   (R=5 wrapping, excluded)
    cubes side=2,3,4
    slabs t=1,2,3   (full other two axes; A fixed, V ∝ t)
  f_S = δS/S_global. f_E = P_flat/M_global = V/N³.
  C_slab PRIMARY. grow = δS(t=3)/δS(t=1) at every α.
          |grow − 3| < |grow − 1|  (volume, not area)
          and grow > 1.5
  C_rho   ρ(f_S, f_E) > 0.90 on the 9-region union, every α
  C_rms   RMS(f_S − f_E) < 0.15 on that union, every α

Not claimed: Clausius, 8πG, de Sitter, MODELS.md.
If C_slab fails, Paper 63 is a packet fact.

Writes ../data/m9_54_sea_enclose.json
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
ALPHAS = (0.01, 0.02, 0.04)
VOL = N**3


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_pbc(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % n, (y + d[1]) % n, (z + d[2]) % n, n)
                    ham[i, j] = ham[j, i] = -1.0
    return ham


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def h_bin(a):
    return float(-a * np.log(a) - (1.0 - a) * np.log(1.0 - a))


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def min_image(a, b, n):
    return min((a - b) % n, (b - a) % n)


def outgoing_area(inside, n):
    area = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if not inside[x, y, z]:
                    continue
                for d in (
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ):
                    xx = (x + d[0]) % n
                    yy = (y + d[1]) % n
                    zz = (z + d[2]) % n
                    if not inside[xx, yy, zz]:
                        area += 1
    return int(area)


def ball_mask(center, radius, n):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                dx = min_image(x, cx, n)
                dy = min_image(y, cy, n)
                dz = min_image(z, cz, n)
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    inside[x, y, z] = True
    return inside


def cube_mask(center, side, n):
    cx, cy, cz = center
    half = side // 2
    inside = np.zeros((n, n, n), dtype=bool)
    for dx in range(side):
        for dy in range(side):
            for dz in range(side):
                inside[
                    (cx - half + dx) % n,
                    (cy - half + dy) % n,
                    (cz - half + dz) % n,
                ] = True
    return inside


def slab_mask(thickness, n):
    inside = np.zeros((n, n, n), dtype=bool)
    inside[:thickness, :, :] = True
    return inside


def region_from_mask(inside, n):
    sl = np.array(
        [
            idx(x, y, z, n)
            for x in range(n)
            for y in range(n)
            for z in range(n)
            if inside[x, y, z]
        ],
        dtype=int,
    )
    return sl, int(inside.sum()), outgoing_area(inside, n)


def main() -> int:
    ham = hop_pbc(N)
    ev, vecs = np.linalg.eigh(ham)
    il, ir = int(np.argmin(ev)), int(np.argmax(ev))
    occ = ev < 0.0
    c0 = vecs[:, occ] @ vecs[:, occ].T
    dC = np.outer(vecs[:, ir], vecs[:, ir]) - np.outer(vecs[:, il], vecs[:, il])
    de1 = np.sum(ham * dC, axis=1)
    specs = (
        [("ball", r) for r in (2, 3, 4)]
        + [("cube", s) for s in (2, 3, 4)]
        + [("slab", t) for t in (1, 2, 3)]
    )
    regions = []
    for kind, p in specs:
        if kind == "ball":
            inside = ball_mask(SRC, p, N)
        elif kind == "cube":
            inside = cube_mask(SRC, p, N)
        else:
            inside = slab_mask(p, N)
        sl, vol, area = region_from_mask(inside, N)
        regions.append(
            {
                "kind": kind,
                "param": p,
                "sl": sl,
                "V": vol,
                "A": area,
                "s0": peschel_s(c0, sl),
                "p1": float(np.sum(de1[sl])),
            }
        )
    s_full0 = peschel_s(c0, np.arange(VOL))
    rows = []
    c_slab = True
    c_rho = True
    c_rms = True
    for alpha in ALPHAS:
        c1 = 0.5 * ((c0 + alpha * dC) + (c0 + alpha * dC).T)
        s_glob = 2.0 * h_bin(alpha)
        s_full1 = peschel_s(c1, np.arange(VOL))
        m_glob = float(np.sum(alpha * de1))
        recs = []
        f_s, f_e, ds_all, aa, vv = [], [], [], [], []
        slab_ds = {}
        for reg in regions:
            ds = peschel_s(c1, reg["sl"]) - reg["s0"]
            p = alpha * reg["p1"]
            fs = ds / s_glob
            fe = p / m_glob
            recs.append(
                {
                    "kind": reg["kind"],
                    "param": reg["param"],
                    "V": reg["V"],
                    "A": reg["A"],
                    "dS": ds,
                    "P_flat": p,
                    "f_S": fs,
                    "f_E": fe,
                    "V_frac": reg["V"] / VOL,
                }
            )
            f_s.append(fs)
            f_e.append(fe)
            ds_all.append(ds)
            aa.append(reg["A"])
            vv.append(reg["V"])
            if reg["kind"] == "slab":
                slab_ds[reg["param"]] = ds
        grow = float(slab_ds[3] / slab_ds[1]) if slab_ds[1] != 0.0 else float("nan")
        rho = pearson(f_s, f_e)
        rms = float(np.sqrt(np.mean((np.asarray(f_s) - np.asarray(f_e)) ** 2)))
        if not (abs(grow - 3.0) < abs(grow - 1.0) and grow > 1.5):
            c_slab = False
        if abs(rho) <= 0.90:
            c_rho = False
        if rms >= 0.15:
            c_rms = False
        rows.append(
            {
                "alpha": alpha,
                "S_global": s_glob,
                "S_full_delta": s_full1 - s_full0,
                "M_global": m_glob,
                "E_L": float(ev[il]),
                "E_R": float(ev[ir]),
                "grow_slab": grow,
                "rho": rho,
                "rms": rms,
                "rho_dS_A": pearson(ds_all, aa),
                "rho_dS_V": pearson(ds_all, vv),
                "regions": recs,
            }
        )
    ok = bool(c_slab and c_rho and c_rms)
    payload = {
        "task": "m9.54_sea_enclose",
        "n_regions": len(regions),
        "rows": rows,
        "C_slab_PRIMARY": c_slab,
        "C_rho": c_rho,
        "C_rms": c_rms,
        "all_gates": ok,
        "verdict": "SEA_ENCLOSURE" if ok else "SEA_NOT_ENCLOSURE",
        "not_claimed": ["Clausius", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_54_sea_enclose.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
