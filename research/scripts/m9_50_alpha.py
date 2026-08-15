#!/usr/bin/env python3
"""M9.50: is κ independent of α, or is it 2h(α)/(α ΔE)?

Papers 46–59 used α=0.02. At that value
S_global = 2h(α) ≈ δS(B_enc) ≈ 0.196.
h(α)=−α log α−(1−α)log(1−α) is not linear in α, so
2h(α)/(α ΔE) runs as −log α. A reusable first-law
constant cannot do that.

PRE-REGISTERED:
  Open hop, N=12, src (6,6,6), σ=1. One eigh.
  α ∈ {0.005, 0.01, 0.02, 0.04, 0.08}.
  Enclosing ball R=3.
  κ(α) = δS(R=3) / P_flat(R=3)
  r_sg(α) = S_global(α) / P_flat(R=3)
  S_global = 2 h(α)
  C_eig   C eigs in [0,1] at every α
  C_lin   PRIMARY. rel range of κ(α):
          (max κ − min κ) / median(κ) < 0.10
  C_sg    r_sg is NOT constant:
          (max r_sg − min r_sg) / median(r_sg) > 0.20
          (S_global/M is not the first-law κ)
  Diagnostic: Pearson(κ, 2h(α)/α) — if C_lin fails and
  this is high, κ was a finite-α artifact.

Not claimed: 8πG, de Sitter, MODELS.md.

Writes ../data/m9_50_alpha.json
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
ALPHAS = (0.005, 0.01, 0.02, 0.04, 0.08)
RADIUS = 3


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


def h_bin(a):
    return float(-a * np.log(a) - (1.0 - a) * np.log(1.0 - a))


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


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
    sl = np.array(
        [
            idx(x, y, z)
            for x in range(N)
            for y in range(N)
            for z in range(N)
            if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= RADIUS * RADIUS
        ],
        dtype=int,
    )
    s0 = peschel_s(c0, sl)
    e0 = np.sum(ham * c0, axis=1)
    rows = []
    kappas, rsgs, twos = [], [], []
    c_eig = True
    for alpha in ALPHAS:
        c1 = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
        c1 = 0.5 * (c1 + c1.T)
        eigs = np.linalg.eigvalsh(c1)
        if eigs.min() < -1e-9 or eigs.max() > 1.0 + 1e-9:
            c_eig = False
        de = np.sum(ham * c1, axis=1) - e0
        ds = peschel_s(c1, sl) - s0
        p = float(np.sum(de[sl]))
        kap = ds / p if abs(p) > 1e-18 else None
        s_glob = 2.0 * h_bin(alpha)
        r_sg = s_glob / p if abs(p) > 1e-18 else None
        two = 2.0 * h_bin(alpha) / alpha
        rows.append(
            {
                "alpha": alpha,
                "deltaS": ds,
                "P_flat": p,
                "kappa": kap,
                "S_global": s_glob,
                "r_sg": r_sg,
                "two_h_over_alpha": two,
                "eig_min": float(eigs.min()),
                "eig_max": float(eigs.max()),
            }
        )
        kappas.append(kap)
        rsgs.append(r_sg)
        twos.append(two)
    kappas = np.asarray(kappas, float)
    rsgs = np.asarray(rsgs, float)
    med_k = float(np.median(kappas))
    med_r = float(np.median(rsgs))
    rel_k = float((np.max(kappas) - np.min(kappas)) / med_k) if med_k else None
    rel_r = float((np.max(rsgs) - np.min(rsgs)) / med_r) if med_r else None
    c_lin = bool(rel_k is not None and rel_k < 0.10)
    c_sg = bool(rel_r is not None and rel_r > 0.20)
    rho_log = pearson(kappas, twos)
    ok = bool(c_eig and c_lin and c_sg)
    payload = {
        "task": "m9.50_alpha",
        "rows": rows,
        "rel_kappa": rel_k,
        "rel_r_sg": rel_r,
        "rho_kappa_vs_2h_over_a": rho_log,
        "C_eig": c_eig,
        "C_lin_PRIMARY": c_lin,
        "C_sg": c_sg,
        "all_gates": ok,
        "verdict": "KAPPA_LINEAR" if ok else "KAPPA_RUNS_WITH_ALPHA",
        "not_claimed": ["8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_50_alpha.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
