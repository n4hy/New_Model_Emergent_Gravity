#!/usr/bin/env python3
"""1d point-source CHM audit. N=160, L=12, source=80, ε=0.08. No import.

Writes ../data/m9_27_audit_1d.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, L, SRC, EPS = 160, 12, 80, 0.08


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
    ham1 = ham.copy()
    ham1[SRC, SRC] += EPS
    ev, vecs = np.linalg.eigh(ham)
    c0 = vecs[:, ev < 0] @ vecs[:, ev < 0].T
    ev, vecs = np.linalg.eigh(ham1)
    c1 = vecs[:, ev < 0] @ vecs[:, ev < 0].T
    de = np.sum(ham1 * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    xs = np.arange(L) - (L - 1) / 2.0
    w = (L / 2.0) ** 2 - xs**2
    ds, pc, pf, pk0, pkmid = [], [], [], [], []
    for s0 in range(0, N - L + 1):
        sl = np.arange(s0, s0 + L)
        block0 = c0[np.ix_(sl, sl)]
        block1 = c1[np.ix_(sl, sl)]

        def S(block):
            z = np.clip(np.linalg.eigvalsh(block), CLIP, 1 - CLIP)
            return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

        def K(block):
            z, u = np.linalg.eigh(block)
            z = np.clip(z, CLIP, 1 - CLIP)
            return (u * np.log((1 - z) / z)) @ u.T

        ds.append(S(block1) - S(block0))
        pc.append(float(np.dot(w, de[sl])))
        pf.append(float(np.sum(de[sl])))
        k0, k1 = K(block0), K(block1)
        dcsl = dc[np.ix_(sl, sl)]
        pk0.append(float(np.sum(k0 * dcsl)))
        pkmid.append(float(np.sum(0.5 * (k0 + k1) * dcsl)))
    ds = np.asarray(ds, float)
    pk0 = np.asarray(pk0, float)
    pkmid = np.asarray(pkmid, float)
    rc, rf = rshape(ds, pc), rshape(ds, pf)
    rho = pearson(ds, pc)
    c2 = bool(rc < rf)
    payload = {
        "task": "m9.27_audit_1d",
        "n": int(len(ds)),
        "rho_CHM": rho,
        "rho_flat": pearson(ds, pf),
        "R_CHM": rc,
        "R_flat": rf,
        "C2": c2,
        "C4": bool(abs(rho) > 0.60),
        "rho_Kvac": pearson(ds, pk0),
        "rho_Kmid": pearson(ds, pkmid),
        "rel_err_Kmid": float(np.max(np.abs(ds - pkmid)) / np.max(np.abs(ds))),
        "verdicts": {
            "C2": "CONFIRMED" if c2 else "REFUTED",
            "Kvac_tracks_dS": "REFUTED" if abs(pearson(ds, pk0)) < 0.50 else "CONFIRMED",
            "Kmid_tracks_dS": "CONFIRMED" if pearson(ds, pkmid) > 0.99 else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_27_audit_1d.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
