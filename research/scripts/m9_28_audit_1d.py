#!/usr/bin/env python3
"""1d fixed-H occupation-transfer audit. N=160, L=12, src=80, σ=2.5, α=0.03.

No import of the solver. Tries to REFUTE C_vac and C2.

Writes ../data/m9_28_audit_1d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, L, SRC, SIG, ALPHA = 160, 12, 80, 2.5, 0.03


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


def rshape(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    m = np.column_stack([x, np.ones(len(x))])
    c, _, _, _ = np.linalg.lstsq(m, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    return float("nan") if den == 0 else float(np.linalg.norm(y - m @ c) / den)


def main() -> int:
    ham = np.zeros((N, N))
    for i in range(N - 1):
        ham[i, i + 1] = ham[i + 1, i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    xs = np.arange(N)
    env = np.exp(-0.5 * ((xs - SRC) / SIG) ** 2)
    stagger = 1.0 - 2.0 * (xs % 2)
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ (stagger * env))
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    c1 = c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))
    c1 = 0.5 * (c1 + c1.T)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    w = (L / 2.0) ** 2 - (np.arange(L) - (L - 1) / 2.0) ** 2
    ds, pc, pf, pk = [], [], [], []
    for s0 in range(0, N - L + 1):
        sl = np.arange(s0, s0 + L)
        b0 = c0[np.ix_(sl, sl)]
        b1 = c1[np.ix_(sl, sl)]

        def S(block):
            z = np.clip(np.linalg.eigvalsh(block), CLIP, 1 - CLIP)
            return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

        def K(block):
            z, u = np.linalg.eigh(block)
            z = np.clip(z, CLIP, 1 - CLIP)
            return (u * np.log((1 - z) / z)) @ u.T

        ds.append(S(b1) - S(b0))
        pc.append(float(np.dot(w, de[sl])))
        pf.append(float(np.sum(de[sl])))
        pk.append(float(np.sum(K(b0) * dc[np.ix_(sl, sl)])))
    ds = np.asarray(ds, float)
    rc, rf = rshape(ds, pc), rshape(ds, pf)
    rho_k = pearson(ds, pk)
    c_vac = bool(abs(rho_k) > 0.95)
    c2 = bool(rc < rf)
    payload = {
        "task": "m9.28_audit_1d",
        "n": int(len(ds)),
        "rho_CHM": pearson(ds, pc),
        "rho_flat": pearson(ds, pf),
        "R_CHM": rc,
        "R_flat": rf,
        "rho_Kvac": rho_k,
        "C_vac": c_vac,
        "C2": c2,
        "verdicts": {
            "C_vac": "CONFIRMED" if c_vac else "REFUTED",
            "C2": "CONFIRMED" if c2 else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_28_audit_1d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_vac and c2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
