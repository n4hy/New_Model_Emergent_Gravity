#!/usr/bin/env python3
"""M9.54 audit. Independent sea construction; tries to REFUTE C_slab.

N=10, periodic hop, own band edges.
α ∈ {0.015, 0.05}.
Regions: balls R=2,3 (no R=4 wrapping), cubes s=2,3, slabs t=1,2,3.

C_slab REFUTE if grow is closer to 1 than to 3, or grow ≤ 1.5.
C_rho  REFUTE if ρ(f_S, f_E) ≤ 0.90 at any α.
C_rms  REFUTE if RMS ≥ 0.15 at any α.

Writes ../data/m9_54_audit_sea.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 10
SRC = (4, 5, 5)
ALPHAS = (0.015, 0.05)
VOL = N**3


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


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
                    if not inside[(x + d[0]) % n, (y + d[1]) % n, (z + d[2]) % n]:
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
    ham = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % N, (y + d[1]) % N, (z + d[2]) % N)
                    ham[i, j] = ham[j, i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    il, ir = int(np.argmin(ev)), int(np.argmax(ev))
    occ = ev < 0.0
    c0 = vecs[:, occ] @ vecs[:, occ].T
    dC = np.outer(vecs[:, ir], vecs[:, ir]) - np.outer(vecs[:, il], vecs[:, il])
    de1 = np.sum(ham * dC, axis=1)
    specs = (
        [("ball", r) for r in (2, 3)]
        + [("cube", s) for s in (2, 3)]
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
    rows = []
    c_slab = True
    c_rho = True
    c_rms = True
    for alpha in ALPHAS:
        c1 = 0.5 * ((c0 + alpha * dC) + (c0 + alpha * dC).T)
        s_glob = 2.0 * h_bin(alpha)
        m_glob = float(np.sum(alpha * de1))
        recs = []
        f_s, f_e = [], []
        slab_ds = {}
        for reg in regions:
            ds = peschel_s(c1, reg["sl"]) - reg["s0"]
            fs = ds / s_glob
            fe = (alpha * reg["p1"]) / m_glob
            recs.append(
                {
                    "kind": reg["kind"],
                    "param": reg["param"],
                    "V": reg["V"],
                    "A": reg["A"],
                    "dS": ds,
                    "f_S": fs,
                    "f_E": fe,
                }
            )
            f_s.append(fs)
            f_e.append(fe)
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
                "M_global": m_glob,
                "grow_slab": grow,
                "rho": rho,
                "rms": rms,
                "regions": recs,
            }
        )
    payload = {
        "task": "m9.54_audit_sea",
        "n_regions": len(regions),
        "rows": rows,
        "C_slab_PRIMARY": c_slab,
        "C_rho": c_rho,
        "C_rms": c_rms,
        "verdicts": {
            "C_slab": "CONFIRMED" if c_slab else "REFUTED",
            "C_rho": "CONFIRMED" if c_rho else "REFUTED",
            "C_rms": "CONFIRMED" if c_rms else "REFUTED",
        },
        "not_claimed": ["Clausius", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_54_audit_sea.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
