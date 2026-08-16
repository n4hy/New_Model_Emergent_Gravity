#!/usr/bin/env python3
"""M9.55: is Paper 64's grow=2.5 a universal area term?

Paper 64: sea-transfer δS is extensive. Slab grow is 2.5,
not 3. A two-term δS = a V + b A would be the remaining
Clausius/horizon handle. In-sample fit is not enough:
the term must predict a held-out shape family.

PRE-REGISTERED:
  N=12, periodic hop, band-edge transfer.
  α ∈ {0.01, 0.02, 0.04}.
  Families (no wrapping):
    balls R=2,3,4
    cubes side=2,3,4
    slabs t=1,2,3
    rods 1×1×L, L=2,3,4,5,6
  Fit (no intercept): δS ≈ a V + b A.
  Volume-only: δS ≈ c V.
  C_fit   in-sample RMS(δS−â)/RMS(δS) < 0.15
  C_gain  two-term RMS < 0.70 × volume-only RMS
  C_loo PRIMARY. leave one family out, predict it:
          ρ(pred, meas) > 0.90 and
          RMS(pred−meas)/RMS(meas) < 0.25
          on every family, every α

Not claimed: Clausius, 1/4G, de Sitter, MODELS.md.
If C_loo fails, grow≠3 is shape junk, not a horizon piece.

Writes ../data/m9_55_twoterm.json
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
FAMILIES = ("ball", "cube", "slab", "rod")


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


def rod_mask(center, length, n):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    for k in range(length):
        inside[cx, cy, (cz + k) % n] = True
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


def specs_for(n):
    return (
        [("ball", r) for r in (2, 3, 4)]
        + [("cube", s) for s in (2, 3, 4)]
        + [("slab", t) for t in (1, 2, 3)]
        + [("rod", ell) for ell in (2, 3, 4, 5, 6)]
    )


def fit_two(vol, area, ds):
    mat = np.column_stack([np.asarray(vol, float), np.asarray(area, float)])
    coef, _, _, _ = np.linalg.lstsq(mat, np.asarray(ds, float), rcond=None)
    return float(coef[0]), float(coef[1])


def fit_vol(vol, ds):
    mat = np.asarray(vol, float).reshape(-1, 1)
    coef, _, _, _ = np.linalg.lstsq(mat, np.asarray(ds, float), rcond=None)
    return float(coef[0])


def rel_rms(pred, meas):
    meas = np.asarray(meas, float)
    pred = np.asarray(pred, float)
    den = float(np.sqrt(np.mean(meas * meas)))
    if den == 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - meas) ** 2)) / den)


def main() -> int:
    ham = hop_pbc(N)
    ev, vecs = np.linalg.eigh(ham)
    il, ir = int(np.argmin(ev)), int(np.argmax(ev))
    occ = ev < 0.0
    c0 = vecs[:, occ] @ vecs[:, occ].T
    dC = np.outer(vecs[:, ir], vecs[:, ir]) - np.outer(vecs[:, il], vecs[:, il])
    regions = []
    for kind, p in specs_for(N):
        if kind == "ball":
            inside = ball_mask(SRC, p, N)
        elif kind == "cube":
            inside = cube_mask(SRC, p, N)
        elif kind == "slab":
            inside = slab_mask(p, N)
        else:
            inside = rod_mask(SRC, p, N)
        sl, vol, area = region_from_mask(inside, N)
        regions.append(
            {
                "kind": kind,
                "param": p,
                "sl": sl,
                "V": vol,
                "A": area,
                "s0": peschel_s(c0, sl),
            }
        )
    rows = []
    c_fit = True
    c_gain = True
    c_loo = True
    for alpha in ALPHAS:
        c1 = 0.5 * ((c0 + alpha * dC) + (c0 + alpha * dC).T)
        recs = []
        for reg in regions:
            ds = peschel_s(c1, reg["sl"]) - reg["s0"]
            recs.append(
                {
                    "kind": reg["kind"],
                    "param": reg["param"],
                    "V": reg["V"],
                    "A": reg["A"],
                    "dS": ds,
                }
            )
        vol = np.array([r["V"] for r in recs], float)
        area = np.array([r["A"] for r in recs], float)
        ds = np.array([r["dS"] for r in recs], float)
        a_coef, b_coef = fit_two(vol, area, ds)
        c_coef = fit_vol(vol, ds)
        pred2 = a_coef * vol + b_coef * area
        pred1 = c_coef * vol
        r_fit = rel_rms(pred2, ds)
        r_vol = rel_rms(pred1, ds)
        if r_fit >= 0.15:
            c_fit = False
        if not (r_fit < 0.70 * r_vol):
            c_gain = False
        loo = []
        for fam in FAMILIES:
            train = [r for r in recs if r["kind"] != fam]
            test = [r for r in recs if r["kind"] == fam]
            aa, bb = fit_two(
                [r["V"] for r in train],
                [r["A"] for r in train],
                [r["dS"] for r in train],
            )
            meas = np.array([r["dS"] for r in test], float)
            pred = aa * np.array([r["V"] for r in test], float) + bb * np.array(
                [r["A"] for r in test], float
            )
            rho = pearson(pred, meas)
            rr = rel_rms(pred, meas)
            loo.append(
                {
                    "held_out": fam,
                    "a": aa,
                    "b": bb,
                    "rho": rho,
                    "rel_rms": rr,
                    "n_test": len(test),
                }
            )
            if abs(rho) <= 0.90 or rr >= 0.25:
                c_loo = False
        rows.append(
            {
                "alpha": alpha,
                "a": a_coef,
                "b": b_coef,
                "c_vol": c_coef,
                "rel_rms_two": r_fit,
                "rel_rms_vol": r_vol,
                "loo": loo,
                "regions": recs,
            }
        )
    ok = bool(c_fit and c_gain and c_loo)
    payload = {
        "task": "m9.55_twoterm",
        "n_regions": len(regions),
        "rows": rows,
        "C_fit": c_fit,
        "C_gain": c_gain,
        "C_loo_PRIMARY": c_loo,
        "all_gates": ok,
        "verdict": "TWO_TERM_AREA" if ok else "TWO_TERM_FAIL",
        "not_claimed": ["Clausius", "1/4G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_55_twoterm.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
