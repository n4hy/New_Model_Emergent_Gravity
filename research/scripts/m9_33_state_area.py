#!/usr/bin/env python3
"""M9.33: state-dependent cut area at fixed H.

Paper 42: hop-area cannot be the second term (δA=0 at fixed
H; when hops move it is energy). The only geometric object
that can respond to matter at fixed hops is built from C.

  A_bond = ∑_{cut NN} |C_ij|     PRIMARY
  A_pur  = ∑_k 4 λ_k (1-λ_k)    spectrum control (may be S in disguise)

PRE-REGISTERED:
  Same occupation transfer as Paper 37. N=12, R=2, 512 balls.
  Packet (6,6,6), σ=1.5, α=0.02 and 0.05. H fixed.
  C_vac    |ρ(δS, Tr(K_vac ΔC))| > 0.95
  C0       max|δS|>1e-6 and max|δA_bond|>1e-8
  C1       Pearson(δS(α), δS(2.5α)) > 0.95
  C_indepP |ρ(δA_bond, P_CHM)| < 0.90     not an energy proxy
  C_indepS |ρ(δA_bond, δS)| < 0.98        not S in disguise
  C_track  |ρ(δS, δA_bond)| > 0.80
  C_eta    PRIMARY. IQR(δS/δA_bond)/|med| < 0.35 on |δA_bond|>1e-8
  A_pur is diagnostic. If |ρ(δA_pur, δS)| > 0.98 it is S.

Not claimed: 8πG, Einstein, de Sitter, MODELS.md.

Writes ../data/m9_33_state_area.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
RADIUS = 2
SRC = (6, 6, 6)
SIGMA = 1.5
ALPHA1 = 0.02
ALPHA2 = 0.05


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


def nn_bonds(n: int):
    bonds = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        bonds.append((i, idx(xx, yy, zz, n)))
    return bonds


def occupation_transfer(ham, n, src, sigma, alpha):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros((n**3,), dtype=float)
    stag = np.zeros((n**3,), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (sigma * sigma))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return c0, 0.5 * (corr + corr.T), int(occ.sum())


def peschel_s_pur(corr, sl):
    ev = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    s = float(-np.sum(ev * np.log(ev) + (1.0 - ev) * np.log(1.0 - ev)))
    pur = float(np.sum(4.0 * ev * (1.0 - ev)))
    return s, pur


def peschel_k(corr, sl):
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def a_bond(corr, inside, bonds):
    acc = 0.0
    for i, j in bonds:
        if (i in inside) != (j in inside):
            acc += abs(corr[i, j])
    return acc


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def residual_ratio(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(y - mat @ coef) / den)


def eta_stats(ds, da, floor=1e-8):
    mask = np.abs(da) > floor
    if int(mask.sum()) == 0:
        return {"n": 0, "median": None, "rel_iqr": None, "pass": False}
    eta = ds[mask] / da[mask]
    med = float(np.median(eta))
    iqr = float(np.percentile(eta, 75) - np.percentile(eta, 25))
    rel = float(iqr / abs(med)) if med != 0.0 else None
    return {
        "n": int(mask.sum()),
        "median": med,
        "rel_iqr": rel,
        "pass": bool(rel is not None and rel < 0.35),
    }


def main() -> int:
    ham = hop_H(N)
    bonds = nn_bonds(N)
    c0, c1, nocc = occupation_transfer(ham, N, SRC, SIGMA, ALPHA1)
    _, c2, _ = occupation_transfer(ham, N, SRC, SIGMA, ALPHA2)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds1, ds2, dbond, dpur, pchm, pk = [], [], [], [], [], []
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
        inside = set(int(i) for i in sl)
        s0, p0 = peschel_s_pur(c0, sl)
        s1, p1 = peschel_s_pur(c1, sl)
        s2, _ = peschel_s_pur(c2, sl)
        ds1.append(s1 - s0)
        ds2.append(s2 - s0)
        dpur.append(p1 - p0)
        dbond.append(a_bond(c1, inside, bonds) - a_bond(c0, inside, bonds))
        pk.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
        s_c = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        s_c += (r2max - rr) * de[idx(x, y, z)]
        pchm.append(s_c)
    ds1 = np.asarray(ds1, float)
    ds2 = np.asarray(ds2, float)
    dbond = np.asarray(dbond, float)
    dpur = np.asarray(dpur, float)
    pchm = np.asarray(pchm, float)
    pk = np.asarray(pk, float)
    rho_k = pearson(ds1, pk)
    rho_sp = pearson(ds1, pchm)
    rho_ab = pearson(ds1, dbond)
    rho_ap = pearson(dbond, pchm)
    rho_as = pearson(dbond, ds1)
    rho_pur_s = pearson(dpur, ds1)
    eta = eta_stats(ds1, dbond)
    c_vac = bool(abs(rho_k) > 0.95)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6 and float(np.max(np.abs(dbond))) > 1e-8)
    c1g = bool(pearson(ds1, ds2) > 0.95)
    c_ind_p = bool(np.isfinite(rho_ap) and abs(rho_ap) < 0.90)
    c_ind_s = bool(np.isfinite(rho_as) and abs(rho_as) < 0.98)
    c_track = bool(np.isfinite(rho_ab) and abs(rho_ab) > 0.80)
    c_eta = bool(eta["pass"])
    gravity = bool(c_vac and c0g and c1g and c_ind_p and c_track and c_eta)
    if not c_vac:
        verdict = "INSTRUMENT_REJECT"
    elif not c0g:
        verdict = "NO_BOND_SIGNAL"
    elif not c_ind_p:
        verdict = "BOND_IS_ENERGY_PROXY"
    elif not c_ind_s:
        verdict = "BOND_IS_ENTROPY"
    elif gravity:
        verdict = "STATE_AREA_CLAUSIUS"
    elif c_track:
        verdict = "STATE_AREA_CORRELATES_NOT_CLAUSIUS"
    else:
        verdict = "STATE_AREA_NOT_CLAUSIUS"
    payload = {
        "task": "m9.33_state_area",
        "n_balls": int(len(centers)),
        "n_occ": int(nocc),
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "max_abs_dA_bond": float(np.max(np.abs(dbond))),
        "max_abs_dA_pur": float(np.max(np.abs(dpur))),
        "rho_eps": pearson(ds1, ds2),
        "rho_Kvac": rho_k,
        "rho_CHM": rho_sp,
        "rho_S_bond": rho_ab,
        "rho_bond_P": rho_ap,
        "rho_pur_S": rho_pur_s,
        "R_bond": residual_ratio(ds1, dbond),
        "R_CHM": residual_ratio(ds1, pchm),
        "eta_bond": eta,
        "C_vac": c_vac,
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C_indepP": c_ind_p,
        "C_indepS": c_ind_s,
        "C_track": c_track,
        "C_eta_PRIMARY": c_eta,
        "pur_is_S": bool(np.isfinite(rho_pur_s) and abs(rho_pur_s) > 0.98),
        "verdict": verdict,
        "not_claimed": ["8pi G", "Einstein equation", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_33_state_area.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if gravity else 1


if __name__ == "__main__":
    raise SystemExit(main())
