#!/usr/bin/env python3
"""M9.47: Fermi-sea vacuum. Negative energy, area-law S. Not Λ.

Paper 56: occupation transfer cannot make negative δe.
The unperturbed sea has E = ∑_{ε<0} ε < 0. If that energy
were a first-law volume source, Gauss would give outward
a ∝ r (the Λ sign). Vacuum entanglement is not a volume
source: it is an area law.

PRE-REGISTERED:
  N=12. Two vacua: open hop, periodic hop. No transfer.
  Centre (6,6,6), balls R=2,3,4,5.
  Open balls: Euclidean. PBC balls: min-image.
  A = number of outgoing NN bonds. V = site count.
  e_i = ∑_j H_ij C0_ij. E_vac = ∑ e_i.
  a_try(R) = −S(R)/R²  (no κ; shape only).
  C_neg   E_vac < 0 on both vacua
  C_area  PRIMARY. |ρ(S,A)| > |ρ(S,V)| on both vacua
  C_ratio S(5)/S(2) closer to A(5)/A(2) than to V(5)/V(2)
          on both vacua
  C_notds slope of |a_try| vs R is not in (0.60, 1.40)
          on both (would be the sea/Λ shape)

Not claimed: 8πG, FGHMV, de Sitter dual, MODELS.md.
Negative E_vac is the usual Fermi-sea energy, not a
measured cosmological constant.

Writes ../data/m9_47_vacuum.json
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
RADII = (2, 3, 4, 5)


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


def slope_of(radii, accs):
    lr = np.log(np.asarray(radii, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([lr, np.ones(len(radii))]), la, rcond=None
    )
    return float(coef[0])


def ball_open(center, radius, n=N):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                    inside[x, y, z] = True
                    sl.append(idx(x, y, z, n))
    area = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if not inside[x, y, z]:
                    continue
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < 0 or yy < 0 or zz < 0 or xx >= n or yy >= n or zz >= n:
                        area += 1
                    elif not inside[xx, yy, zz]:
                        area += 1
    return np.array(sl, dtype=int), int(inside.sum()), area


def ball_pbc(center, radius, n=N):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                dx = min((x - cx) % n, (cx - x) % n)
                dy = min((y - cy) % n, (cy - y) % n)
                dz = min((z - cz) % n, (cz - z) % n)
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    inside[x, y, z] = True
                    sl.append(idx(x, y, z, n))
    area = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if not inside[x, y, z]:
                    continue
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    xx, yy, zz = (x + d[0]) % n, (y + d[1]) % n, (z + d[2]) % n
                    if not inside[xx, yy, zz]:
                        area += 1
    return np.array(sl, dtype=int), int(inside.sum()), area


def vacuum_scan(ham, ball_fn):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    c0 = vecs[:, occ] @ vecs[:, occ].T
    e_site = np.sum(ham * c0, axis=1)
    e_vac = float(np.sum(e_site))
    mean_abs = float(np.mean(np.abs(e_site)))
    unif = float(np.std(e_site) / mean_abs) if mean_abs else None
    ss, vv, aa = [], [], []
    for rad in RADII:
        sl, v, a = ball_fn(SRC, rad)
        ss.append(peschel_s(c0, sl))
        vv.append(v)
        aa.append(a)
    ss, vv, aa = map(np.asarray, (ss, vv, aa))
    a_try = -ss / np.asarray(RADII, float) ** 2
    return {
        "E_vac": e_vac,
        "e_unif": unif,
        "S": ss.tolist(),
        "V": vv.tolist(),
        "A": aa.tolist(),
        "rho_SA": pearson(ss, aa),
        "rho_SV": pearson(ss, vv),
        "grow_S": float(ss[-1] / ss[0]),
        "grow_A": float(aa[-1] / aa[0]),
        "grow_V": float(vv[-1] / vv[0]),
        "a_try": a_try.tolist(),
        "slope_try": slope_of(RADII, a_try),
    }


def nearer(target, a, b):
    return abs(target - a) < abs(target - b)


def main() -> int:
    open_row = vacuum_scan(hop_open(N), ball_open)
    pbc_row = vacuum_scan(hop_pbc(N), ball_pbc)
    c_neg = bool(open_row["E_vac"] < 0.0 and pbc_row["E_vac"] < 0.0)
    c_area = bool(
        abs(open_row["rho_SA"]) > abs(open_row["rho_SV"])
        and abs(pbc_row["rho_SA"]) > abs(pbc_row["rho_SV"])
    )
    c_ratio = bool(
        nearer(open_row["grow_S"], open_row["grow_A"], open_row["grow_V"])
        and nearer(pbc_row["grow_S"], pbc_row["grow_A"], pbc_row["grow_V"])
    )
    c_notds = bool(
        not (0.60 < open_row["slope_try"] < 1.40)
        and not (0.60 < pbc_row["slope_try"] < 1.40)
    )
    ok = bool(c_neg and c_area and c_ratio and c_notds)
    payload = {
        "task": "m9.47_vacuum",
        "open": open_row,
        "pbc": pbc_row,
        "C_neg": c_neg,
        "C_area_PRIMARY": c_area,
        "C_ratio": c_ratio,
        "C_notds": c_notds,
        "all_gates": ok,
        "verdict": "VACUUM_AREA_NOT_LAMBDA" if ok else "VACUUM_FAIL",
        "not_claimed": [
            "8pi G",
            "FGHMV",
            "de Sitter dual",
            "MODELS.md",
            "E_vac is a measured Lambda",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_47_vacuum.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
