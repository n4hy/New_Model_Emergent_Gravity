#!/usr/bin/env python3
"""M9.36 audit. N=10, R=3, α=0.03, σ=0.9.

One source (5,5,5) vs two (4,5,5)+(6,5,5). Tries to REFUTE
C_univ: |med κ1 − med κ2| / mean|med| < 0.15.

Writes ../data/m9_36_audit_kappa.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, ALPHA = 10, 3, 0.9, 0.03


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


def main() -> int:
    vol = N**3
    ham = np.zeros((vol, vol))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        j = idx(xx, yy, zz)
                        ham[i, j] = ham[j, i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T

    def pack(src):
        env = np.zeros(vol)
        stag = np.zeros(vol)
        sx, sy, sz = src
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                    env[i] = np.exp(-0.5 * rr / (SIG * SIG))
                    stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
        left, right = uo @ (uo.T @ env), uu @ (uu.T @ stag)
        return left / np.linalg.norm(left), right / np.linalg.norm(right)

    def onorm(v1, v2):
        e1 = v1 / np.linalg.norm(v1)
        v2 = v2 - e1 * np.dot(e1, v2)
        return e1, v2 / np.linalg.norm(v2)

    def run(sources):
        corr = c0.copy()
        if len(sources) == 1:
            left, right = pack(sources[0])
            corr = corr + ALPHA * (np.outer(right, right) - np.outer(left, left))
        else:
            l1, r1 = pack(sources[0])
            l2, r2 = pack(sources[1])
            l1, l2 = onorm(l1, l2)
            r1, r2 = onorm(r1, r2)
            corr = corr + ALPHA * (
                np.outer(r1, r1) - np.outer(l1, l1) + np.outer(r2, r2) - np.outer(l2, l2)
            )
        corr = 0.5 * (corr + corr.T)
        de = np.sum(ham * corr, axis=1) - np.sum(ham * c0, axis=1)
        r2max = RADIUS * RADIUS

        def S(c, sl):
            z = np.clip(np.linalg.eigvalsh(c[np.ix_(sl, sl)]), CLIP, 1 - CLIP)
            return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

        def inside(src, c):
            return sum((src[k] - c[k]) ** 2 for k in range(3)) <= r2max

        kap = []
        ds_in, pf_in = [], []
        for cx in range(RADIUS, N - RADIUS):
            for cy in range(RADIUS, N - RADIUS):
                for cz in range(RADIUS, N - RADIUS):
                    if not all(inside(s, (cx, cy, cz)) for s in sources):
                        continue
                    sl, s_f = [], 0.0
                    for x in range(N):
                        for y in range(N):
                            for z in range(N):
                                rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                                if rr <= r2max:
                                    sl.append(idx(x, y, z))
                                    s_f += de[idx(x, y, z)]
                    sl = np.array(sl, dtype=int)
                    dsv = S(corr, sl) - S(c0, sl)
                    if abs(s_f) > 1e-6:
                        kap.append(dsv / s_f)
                        ds_in.append(dsv)
                        pf_in.append(s_f)
        kap = np.asarray(kap, float)
        med = float(np.median(kap))
        iqr = float(np.percentile(kap, 75) - np.percentile(kap, 25))
        return {
            "n": int(len(kap)),
            "median": med,
            "rel_iqr": float(iqr / abs(med)) if med else None,
            "rho": pearson(ds_in, pf_in),
        }

    one = run([(5, 5, 5)])
    two = run([(4, 5, 5), (6, 5, 5)])
    mean_abs = 0.5 * (abs(one["median"]) + abs(two["median"]))
    rel = abs(one["median"] - two["median"]) / mean_abs if mean_abs else None
    c_univ = bool(
        rel is not None
        and rel < 0.15
        and one["rel_iqr"] is not None
        and one["rel_iqr"] < 0.35
        and two["rel_iqr"] is not None
        and two["rel_iqr"] < 0.35
    )
    payload = {
        "task": "m9.36_audit_kappa",
        "one": one,
        "two": two,
        "rel_median": rel,
        "C_univ": c_univ,
        "verdicts": {"C_univ": "CONFIRMED" if c_univ else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_36_audit_kappa.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c_univ else 1


if __name__ == "__main__":
    raise SystemExit(main())
