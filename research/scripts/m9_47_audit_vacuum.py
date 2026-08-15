#!/usr/bin/env python3
"""M9.47 audit. N=10, centre (5,5,5), open + PBC vacua.

Tries to REFUTE C_area and C_neg.

Writes ../data/m9_47_audit_vacuum.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SRC = 10, (5, 5, 5)
RADII = (2, 3, 4)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


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


def scan(pbc: bool):
    ham = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if pbc:
                        j = idx(xx % N, yy % N, zz % N)
                        ham[i, j] = ham[j, i] = -1.0
                    elif xx < N and yy < N and zz < N:
                        ham[i, idx(xx, yy, zz)] = ham[idx(xx, yy, zz), i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    c0 = vecs[:, occ] @ vecs[:, occ].T
    e_vac = float(np.sum(ham * c0))
    ss, vv, aa = [], [], []
    sx, sy, sz = SRC
    for rad in RADII:
        inside = np.zeros((N, N, N), dtype=bool)
        sl = []
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if pbc:
                        dx = min((x - sx) % N, (sx - x) % N)
                        dy = min((y - sy) % N, (sy - y) % N)
                        dz = min((z - sz) % N, (sz - z) % N)
                        hit = dx * dx + dy * dy + dz * dz <= rad * rad
                    else:
                        hit = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= rad * rad
                    if hit:
                        inside[x, y, z] = True
                        sl.append(idx(x, y, z))
        sl = np.array(sl, dtype=int)
        area = 0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if not inside[x, y, z]:
                        continue
                    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        xx, yy, zz = x + d[0], y + d[1], z + d[2]
                        if pbc:
                            xx, yy, zz = xx % N, yy % N, zz % N
                            if not inside[xx, yy, zz]:
                                area += 1
                        elif xx < 0 or yy < 0 or zz < 0 or xx >= N or yy >= N or zz >= N:
                            area += 1
                        elif not inside[xx, yy, zz]:
                            area += 1
        ss.append(peschel_s(c0, sl))
        vv.append(int(inside.sum()))
        aa.append(area)
    ss, vv, aa = map(np.asarray, (ss, vv, aa))
    return {
        "E_vac": e_vac,
        "rho_SA": pearson(ss, aa),
        "rho_SV": pearson(ss, vv),
        "S": ss.tolist(),
        "A": aa.tolist(),
        "V": vv.tolist(),
    }


def main() -> int:
    op, pb = scan(False), scan(True)
    c_neg = bool(op["E_vac"] < 0.0 and pb["E_vac"] < 0.0)
    c_area = bool(abs(op["rho_SA"]) > abs(op["rho_SV"]) and abs(pb["rho_SA"]) > abs(pb["rho_SV"]))
    payload = {
        "task": "m9.47_audit_vacuum",
        "open": op,
        "pbc": pb,
        "C_neg": c_neg,
        "C_area": c_area,
        "verdicts": {
            "C_neg": "CONFIRMED" if c_neg else "REFUTED",
            "C_area": "CONFIRMED" if c_area else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_47_audit_vacuum.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_neg and c_area) else 1


if __name__ == "__main__":
    raise SystemExit(main())
