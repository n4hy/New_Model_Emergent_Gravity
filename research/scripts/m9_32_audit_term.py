#!/usr/bin/env python3
"""M9.32 audit. N=10, R=2. Geometry-only partial ρ. No import.

Tries to REFUTE C_partial: |partial ρ(δS, δA | P)| > 0.50.

Writes ../data/m9_32_audit_term.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, RADIUS, SIG, EPS = 10, 2, 1.5, 0.03
SRC = (4, 5, 5)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if d == 0 else float(np.dot(a, b) / d)


def main() -> int:
    vol = N**3
    phi = np.zeros(vol)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                phi[idx(x, y, z)] = np.exp(-0.5 * rr / (SIG * SIG))

    def ham_tm(eps):
        ham = np.zeros((vol, vol))
        tm = {}
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        xx, yy, zz = x + d[0], y + d[1], z + d[2]
                        if xx < N and yy < N and zz < N:
                            j = idx(xx, yy, zz)
                            t = -(1.0 + eps * 0.5 * (phi[i] + phi[j]))
                            ham[i, j] = ham[j, i] = t
                            tm[(min(i, j), max(i, j))] = t
        return ham, tm

    def occ(ham):
        ev, vecs = np.linalg.eigh(ham)
        fill = ev < 0.0
        return vecs[:, fill] @ vecs[:, fill].T

    def S(c, sl):
        z = np.clip(np.linalg.eigvalsh(c[np.ix_(sl, sl)]), CLIP, 1 - CLIP)
        return float(-np.sum(z * np.log(z) + (1 - z) * np.log(1 - z)))

    def aface(sl, tm):
        inside = set(int(i) for i in sl)
        acc = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    if i not in inside:
                        continue
                    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        xx, yy, zz = x + d[0], y + d[1], z + d[2]
                        if not (0 <= xx < N and 0 <= yy < N and 0 <= zz < N):
                            acc += 1.0
                            continue
                        j = idx(xx, yy, zz)
                        if j not in inside:
                            t = tm[(min(i, j), max(i, j))]
                            acc += 1.0 / (t * t)
        return acc

    h0, t0 = ham_tm(0.0)
    h1, t1 = ham_tm(EPS)
    c0, c1 = occ(h0), occ(h1)
    de = np.sum(h1 * c1, axis=1) - np.sum(h0 * c0, axis=1)
    r2max = RADIUS * RADIUS
    ds, da, pc = [], [], []
    for cx in range(RADIUS, N - RADIUS):
        for cy in range(RADIUS, N - RADIUS):
            for cz in range(RADIUS, N - RADIUS):
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
                ds.append(S(c1, sl) - S(c0, sl))
                da.append(aface(sl, t1) - aface(sl, t0))
                s = 0.0
                for x in range(N):
                    for y in range(N):
                        for z in range(N):
                            rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                            if rr <= r2max:
                                s += (r2max - rr) * de[idx(x, y, z)]
                pc.append(s)
    ds, da, pc = map(np.asarray, (ds, da, pc))
    rsa, rsp, rap = pearson(ds, da), pearson(ds, pc), pearson(da, pc)
    den = np.sqrt((1 - rsp * rsp) * (1 - rap * rap))
    part = float("nan") if den == 0 else float((rsa - rsp * rap) / den)
    c_part = bool(np.isfinite(part) and abs(part) > 0.50)
    colin = bool(abs(rap) > 0.90)
    payload = {
        "task": "m9.32_audit_term",
        "n": int(len(ds)),
        "rho_S_A": rsa,
        "rho_S_P": rsp,
        "rho_A_P": rap,
        "partial_rho_S_A_given_P": part,
        "C_partial_raw": c_part,
        "C_partial_scored": bool(c_part and not colin),
        "verdicts": {
            "C_partial_independence": (
                "CONFIRMED" if (c_part and not colin) else "REFUTED"
            ),
            "collinear": "YES" if colin else "NO",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_32_audit_term.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_part and not colin) else 1


if __name__ == "__main__":
    raise SystemExit(main())
