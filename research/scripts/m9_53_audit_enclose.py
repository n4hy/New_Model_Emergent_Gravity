#!/usr/bin/env python3
"""M9.53 audit. Independent construction; tries to REFUTE C_rho.

N=10, R=3, src (4,5,5), σ=0.9, α ∈ {0.01, 0.06}.
Own 64 centres. Own Hamiltonian. Own transfer.

C_rho REFUTE if ρ(f_S, f_E) ≤ 0.95 at any α.
C_rms REFUTE if RMS ≥ 0.10 at any α.
C_enc REFUTE if well-inside median|f−1| ≥ 0.05.

Writes ../data/m9_53_audit_enclose.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG = 10, 3, 0.9
SRC = (4, 5, 5)
ALPHAS = (0.01, 0.06)


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


def main() -> int:
    ham = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        ham[i, idx(xx, yy, zz)] = ham[idx(xx, yy, zz), i] = -1.0
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
                env[i] = np.exp(-0.5 * rr / (SIG * SIG))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    c0 = uo @ uo.T
    dC = np.outer(right, right) - np.outer(left, left)
    de1 = np.sum(ham * dC, axis=1)
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    r2max = RADIUS * RADIUS
    slices, offsets, inside = [], [], []
    for cx, cy, cz in centers:
        sl = np.array(
            [
                idx(x, y, z)
                for x in range(N)
                for y in range(N)
                for z in range(N)
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2max
            ],
            dtype=int,
        )
        slices.append(sl)
        off = (sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2
        offsets.append(off)
        inside.append(off <= r2max)
    inside = np.asarray(inside, bool)
    offsets = np.asarray(offsets, float)
    s0 = [peschel_s(c0, sl) for sl in slices]
    p1 = np.array([float(np.sum(de1[sl])) for sl in slices])
    centered = [i for i, c in enumerate(centers) if c == SRC]
    rows = []
    c_rho = True
    c_rms = True
    c_enc = True
    for alpha in ALPHAS:
        c1 = 0.5 * ((c0 + alpha * dC) + (c0 + alpha * dC).T)
        ds = np.array([peschel_s(c1, sl) - s0[i] for i, sl in enumerate(slices)])
        p = alpha * p1
        s_glob = 2.0 * h_bin(alpha)
        m_glob = float(np.sum(alpha * de1))
        f_s = ds / s_glob
        f_e = p / m_glob
        rho = pearson(f_s, f_e)
        rms = float(np.sqrt(np.mean((f_s - f_e) ** 2)))
        well = inside & (np.abs(p) > 1e-6)
        med_s = float(np.median(np.abs(f_s[well] - 1.0)))
        med_e = float(np.median(np.abs(f_e[well] - 1.0)))
        if abs(rho) <= 0.95:
            c_rho = False
        if rms >= 0.10:
            c_rms = False
        if med_s >= 0.05 or med_e >= 0.05:
            c_enc = False
        rec = {
            "alpha": alpha,
            "S_global": s_glob,
            "M_global": m_glob,
            "rho": rho,
            "rms": rms,
            "n_well": int(np.sum(well)),
            "med_abs_fS_minus_1": med_s,
            "med_abs_fE_minus_1": med_e,
        }
        if centered:
            i0 = centered[0]
            rec["centered_fS"] = float(f_s[i0])
            rec["centered_fE"] = float(f_e[i0])
            rec["centered_offset2"] = float(offsets[i0])
        rows.append(rec)
    payload = {
        "task": "m9.53_audit_enclose",
        "n_balls": len(centers),
        "src": list(SRC),
        "sigma": SIG,
        "rows": rows,
        "C_rho_PRIMARY": c_rho,
        "C_rms": c_rms,
        "C_enc": c_enc,
        "verdicts": {
            "C_rho": "CONFIRMED" if c_rho else "REFUTED",
            "C_rms": "CONFIRMED" if c_rms else "REFUTED",
            "C_enc": "CONFIRMED" if c_enc else "REFUTED",
        },
        "not_claimed": ["Clausius", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_53_audit_enclose.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
