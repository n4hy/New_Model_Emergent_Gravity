#!/usr/bin/env python3
"""M9.34: two-source pair. Gauss (enclosed) vs CHM boost.

An opposite-energy dipole is impossible on the Fermi-sea
vacuum: every allowed occupation swap raises ⟨H⟩. Two
positive PH packets are two masses. On balls that contain
both, P_flat is nearly constant (enclosed energy) while
P_CHM varies with where the two sit in the boost weight.

PRE-REGISTERED:
  N=12, R=2, 512 balls. H fixed.
  Sources (5,6,6) and (7,6,6), σ=1.0.
  Orthonormal L± in occupied, R± in unoccupied.
  C = C0 + α (Δ+ + Δ−), Δ = |R⟩⟨R| − |L⟩⟨L|.
  α=0.02 and 0.05.
  C_vac  |ρ(δS, Tr(K_vac ΔC))| > 0.95
  C0     max|δS| > 1e-6
  C1     Pearson(δS(α), δS(2.5α)) > 0.95
  C2     PRIMARY on ALL balls. R_CHM < R_flat
  C2b    PRIMARY on both-inside subset (n≥10).
         R_CHM < R_flat. This is Gauss vs boost.
  C4     |ρ(δS, P_CHM)| > 0.60 on all balls

Not claimed: 8πG, Einstein, de Sitter, MODELS.md.

Writes ../data/m9_34_dipole.json
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
A = (5, 6, 6)
B = (7, 6, 6)
SIGMA = 1.0
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


def raw_packet(uo, uu, n, src, sigma):
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
    return left, right


def orthonormalize(v1, v2):
    n1 = np.linalg.norm(v1)
    if n1 < 1e-14:
        raise RuntimeError("packet vanished")
    e1 = v1 / n1
    v2 = v2 - e1 * np.dot(e1, v2)
    n2 = np.linalg.norm(v2)
    if n2 < 1e-14:
        raise RuntimeError("packets linearly dependent")
    return e1, v2 / n2


def two_source(ham, n, src_a, src_b, sigma, alpha):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    la, ra = raw_packet(uo, uu, n, src_a, sigma)
    lb, rb = raw_packet(uo, uu, n, src_b, sigma)
    la, lb = orthonormalize(la, lb)
    ra, rb = orthonormalize(ra, rb)
    c0 = uo @ uo.T
    corr = (
        c0
        + alpha * (np.outer(ra, ra) - np.outer(la, la))
        + alpha * (np.outer(rb, rb) - np.outer(lb, lb))
    )
    return c0, 0.5 * (corr + corr.T), int(occ.sum())


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr, sl):
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def in_ball(src, center, radius):
    return sum((src[k] - center[k]) ** 2 for k in range(3)) <= radius * radius


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
    if len(y) < 3:
        return float("nan")
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(y - mat @ coef) / den)


def main() -> int:
    ham = hop_H(N)
    c0, c1, nocc = two_source(ham, N, A, B, SIGMA, ALPHA1)
    _, c2, _ = two_source(ham, N, A, B, SIGMA, ALPHA2)
    ev1 = np.linalg.eigvalsh(c1)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds1, ds2, pchm, pflat, plin, pk, both = [], [], [], [], [], [], []
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
        ds1.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        ds2.append(peschel_s(c2, sl) - peschel_s(c0, sl))
        pk.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
        s_c = s_l = s_f = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        e = de[idx(x, y, z)]
                        s_c += (r2max - rr) * e
                        s_l += (RADIUS - np.sqrt(rr)) * e
                        s_f += e
        pchm.append(s_c)
        plin.append(s_l)
        pflat.append(s_f)
        both.append(in_ball(A, (cx, cy, cz), RADIUS) and in_ball(B, (cx, cy, cz), RADIUS))
    ds1, ds2 = np.asarray(ds1, float), np.asarray(ds2, float)
    pchm, pflat, plin = map(np.asarray, (pchm, pflat, plin))
    pk = np.asarray(pk, float)
    both = np.asarray(both, bool)
    r_chm = residual_ratio(ds1, pchm)
    r_flat = residual_ratio(ds1, pflat)
    r_lin = residual_ratio(ds1, plin)
    n_both = int(both.sum())
    if n_both >= 10:
        r_chm_b = residual_ratio(ds1[both], pchm[both])
        r_flat_b = residual_ratio(ds1[both], pflat[both])
        rho_c_b = pearson(ds1[both], pchm[both])
        rho_f_b = pearson(ds1[both], pflat[both])
        std_f_b = float(np.std(pflat[both]))
        std_c_b = float(np.std(pchm[both]))
    else:
        r_chm_b = r_flat_b = rho_c_b = rho_f_b = std_f_b = std_c_b = float("nan")
    c_vac = bool(abs(pearson(ds1, pk)) > 0.95)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6)
    c1g = bool(pearson(ds1, ds2) > 0.95)
    c2g = bool(r_chm < r_flat)
    c2b = bool(n_both >= 10 and r_chm_b < r_flat_b)
    c4g = bool(abs(pearson(ds1, pchm)) > 0.60)
    ok = bool(c_vac and c0g and c1g and c2g and c2b and c4g)
    if not c_vac:
        verdict = "INSTRUMENT_REJECT"
    elif ok:
        verdict = "PAIR_CHM_BEATS_GAUSS"
    elif c2g and not c2b:
        verdict = "CHM_ALL_GAUSS_SUBSET"
    elif not c2g:
        verdict = "PAIR_FLAT"
    else:
        verdict = "PAIR_MIXED"
    payload = {
        "task": "m9.34_two_source",
        "n_balls": int(len(centers)),
        "n_both_inside": n_both,
        "n_occ": int(nocc),
        "c_eig_min": float(np.min(ev1)),
        "c_eig_max": float(np.max(ev1)),
        "sum_de": float(np.sum(de)),
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "rho_eps": pearson(ds1, ds2),
        "rho_Kvac": pearson(ds1, pk),
        "rho_CHM": pearson(ds1, pchm),
        "rho_flat": pearson(ds1, pflat),
        "R_CHM": r_chm,
        "R_lin": r_lin,
        "R_flat": r_flat,
        "both_inside": {
            "n": n_both,
            "rho_CHM": rho_c_b,
            "rho_flat": rho_f_b,
            "R_CHM": r_chm_b,
            "R_flat": r_flat_b,
            "std_P_flat": std_f_b,
            "std_P_CHM": std_c_b,
        },
        "C_vac": c_vac,
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C2_PRIMARY_all": c2g,
        "C2b_PRIMARY_both_inside": c2b,
        "C4_tracks": c4g,
        "all_gates": ok,
        "verdict": verdict,
        "not_claimed": ["8pi G", "Einstein equation", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_34_dipole.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
