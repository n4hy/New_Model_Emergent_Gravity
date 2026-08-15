#!/usr/bin/env python3
"""M9.43 audit. N=10, source (4,5,5), σ=0.9, α=0.03, m ∈ {0.0, 0.40}.

Own staggered H, own occupation transfer. Tries to REFUTE C_univ.

Writes ../data/m9_43_audit_diamond.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SRC, SIG, ALPHA = 10, (4, 5, 5), 0.9, 0.03
MASSES = (0.0, 0.40)
RADII = (2, 3, 4)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def main() -> int:
    kappas = {}
    rows = []
    for mass in MASSES:
        ham = np.zeros((N**3, N**3))
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    ham[i, i] = mass * (1.0 if (x + y + z) % 2 == 0 else -1.0)
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
        left = left / np.linalg.norm(left)
        right = right / np.linalg.norm(right)
        c0 = uo @ uo.T
        c1 = c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))
        c1 = 0.5 * (c1 + c1.T)
        de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
        m_glob = float(np.sum(de))
        r_enc, kap = None, None
        scan = []
        for rad in RADII:
            sl = np.array(
                [
                    idx(x, y, z)
                    for x in range(N)
                    for y in range(N)
                    for z in range(N)
                    if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= rad * rad
                ],
                dtype=int,
            )
            ds = peschel_s(c1, sl) - peschel_s(c0, sl)
            p = float(np.sum(de[sl]))
            encl = abs(p / m_glob) if m_glob else 0.0
            k = (ds / p) if abs(p) > 1e-18 else None
            scan.append({"R": rad, "deltaS": ds, "P_flat": p, "kappa": k, "enclose": encl})
            if r_enc is None and encl > 0.95:
                r_enc, kap = rad, k
        kappas[str(mass)] = kap
        rows.append({"m": mass, "M_global": m_glob, "R_enc": r_enc, "kappa": kap, "scan": scan})
    k0, k1 = kappas["0.0"], kappas["0.4"]
    univ = abs(k1 / k0 - 1.0) if (k0 and k1) else None
    c_univ = bool(univ is not None and univ < 0.15)
    c_encl = bool(all(r["R_enc"] is not None for r in rows))
    payload = {
        "task": "m9.43_audit_diamond",
        "kappa_of_m": kappas,
        "univ_rel": univ,
        "rows": rows,
        "C_encl": c_encl,
        "C_univ": c_univ,
        "verdicts": {
            "C_univ": "CONFIRMED" if c_univ else "REFUTED",
            "C_encl": "CONFIRMED" if c_encl else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_43_audit_diamond.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_univ and c_encl) else 1


if __name__ == "__main__":
    raise SystemExit(main())
