#!/usr/bin/env python3
"""M9.52: which predictor of δS wins as α runs — P_flat, P_CHM, Tr(K ΔC)?

Paper 37: 3d balls, Tr(K_vac ΔC) and CHM. Paper 45: well-inside,
enclosed energy. Paper 60: finite δS on an enclosing ball is
2h(α), not linear. This run puts the three predictors on the
same 216 balls at three α.

PRE-REGISTERED:
  N=12, R=3, 216 centres. Open hop. Packet (6,6,6), σ=1.
  α ∈ {0.005, 0.02, 0.08}.
  P_flat = ∑_B δe
  P_CHM  = ∑_B (R²−r²) δe
  T_K    = Tr(K_vac ΔC) on B  (K from C0; ΔC ∝ α)
  Well-inside: source in the ball and |P_flat|>1e-6.
  Winner = largest |Pearson| among {P_flat, P_CHM, T_K}.
  C_track  max|ρ| > 0.90 on all-balls at every α
  C_win    PRIMARY. report winner on all-balls and on
           well-inside, each α. C_flip if the all-ball
           winner changes with α.

Not claimed: 8πG, Einstein, de Sitter, MODELS.md.

Writes ../data/m9_52_predictor.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
RADIUS = 3
SRC = (6, 6, 6)
SIGMA = 1.0
ALPHAS = (0.005, 0.02, 0.08)


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H(n: int) -> np.ndarray:
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


def peschel_k(corr, sl):
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def main() -> int:
    ham = hop_H(N)
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
    dC = np.outer(right, right) - np.outer(left, left)
    e0 = np.sum(ham * c0, axis=1)
    # site energy of the α=1 update (linear)
    # de(α) = α * de1, de1_i = ∑_j H_ij dC_ij
    de1 = np.sum(ham * dC, axis=1)

    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    r2max = RADIUS * RADIUS
    slices, chm_w, inside = [], [], []
    for cx, cy, cz in centers:
        sl = []
        wsum = []
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        sl.append(idx(x, y, z))
                        wsum.append(r2max - rr)
        sl = np.array(sl, dtype=int)
        slices.append(sl)
        chm_w.append(np.asarray(wsum, float))
        inside.append((sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2 <= r2max)
    inside = np.asarray(inside, bool)

    s0 = [peschel_s(c0, sl) for sl in slices]
    k0 = [peschel_k(c0, sl) for sl in slices]
    tk1 = []
    pflat1 = []
    pchm1 = []
    for sl, w, kv in zip(slices, chm_w, k0):
        dc_b = dC[np.ix_(sl, sl)]
        tk1.append(float(np.sum(kv * dc_b)))
        pflat1.append(float(np.sum(de1[sl])))
        pchm1.append(float(np.sum(w * de1[sl])))
    tk1 = np.asarray(tk1)
    pflat1 = np.asarray(pflat1)
    pchm1 = np.asarray(pchm1)

    rows = []
    winners_all = []
    c_track = True
    for alpha in ALPHAS:
        c1 = 0.5 * ((c0 + alpha * dC) + (c0 + alpha * dC).T)
        ds = np.array([peschel_s(c1, sl) - s0[i] for i, sl in enumerate(slices)])
        # predictors scale with α except finite δS
        preds = {
            "P_flat": alpha * pflat1,
            "P_CHM": alpha * pchm1,
            "T_K": alpha * tk1,
        }
        rhos = {name: pearson(ds, pred) for name, pred in preds.items()}
        winner = max(rhos, key=lambda n: abs(rhos[n]))
        winners_all.append(winner)
        if abs(rhos[winner]) <= 0.90:
            c_track = False
        well = inside & (np.abs(preds["P_flat"]) > 1e-6)
        rhos_w = {name: pearson(ds[well], pred[well]) for name, pred in preds.items()}
        winner_w = max(rhos_w, key=lambda n: abs(rhos_w[n]))
        rows.append(
            {
                "alpha": alpha,
                "n_well": int(np.sum(well)),
                "rho_all": rhos,
                "winner_all": winner,
                "rho_well": rhos_w,
                "winner_well": winner_w,
            }
        )
    c_flip = bool(len(set(winners_all)) > 1)
    payload = {
        "task": "m9.52_predictor",
        "n_balls": len(centers),
        "rows": rows,
        "winners_all": winners_all,
        "C_track": c_track,
        "C_flip_PRIMARY": c_flip,
        "verdict": "WINNER_FLIPS" if c_flip else "WINNER_STABLE",
        "not_claimed": ["8pi G", "Einstein", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_52_predictor.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c_track else 1


if __name__ == "__main__":
    raise SystemExit(main())
