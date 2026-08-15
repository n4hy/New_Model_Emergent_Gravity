#!/usr/bin/env python3
"""M9.27: 1d point-source first law (CHM is a theorem here).

If this instrument is valid, CHM must beat flat on a 1d massless
interval. If flat wins here too, Paper 35's 3d result is an
instrument failure (wrong δe), not a 3d theorem.

PRE-REGISTERED:
  Chain N=200, hop -1. Intervals of length L=16 at every
  start 0..184 (185 intervals). Source: ε at site 100.
  ε=0.05 and 0.10. E<0; half-fill if n_occ flips.
  e_i = ∑_j H_ij C_ij.
  P_CHM = ∑_{i in I} ((L/2)^2 - x_i^2) δe_i
  P_flat = ∑_{i in I} δe_i
  C0  max|δS| > 1e-6.
  C1  Pearson(δS(ε), δS(2ε)) > 0.95.
  C2  PRIMARY. R_shape(δS, P_CHM) < R_shape(δS, P_flat).
  C4  |ρ(δS, P_CHM)| > 0.60.

Diagnostics (not gates): ρ(δS, Tr(K_vac ΔC)) and
ρ(δS, Tr(K_mid ΔC)). These ask whether the vacuum first
law identity holds for a point Hamiltonian source.

Writes ../data/m9_27_1d_point_chm.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 200
L = 16
SRC = 100
EPS1 = 0.05
EPS2 = 0.10


def hop_H(n: int) -> np.ndarray:
    ham = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        ham[i, i + 1] = ham[i + 1, i] = -1.0
    return ham


def occupy(ham: np.ndarray, half: bool) -> tuple[np.ndarray, int]:
    ev, vecs = np.linalg.eigh(ham)
    if half:
        nocc = ham.shape[0] // 2
        return vecs[:, :nocc] @ vecs[:, :nocc].T, nocc
    filled = ev < 0.0
    return vecs[:, filled] @ vecs[:, filled].T, int(filled.sum())


def site_energy(ham: np.ndarray, corr: np.ndarray) -> np.ndarray:
    return np.sum(ham * corr, axis=1)


def peschel_s(corr: np.ndarray, sl: np.ndarray) -> float:
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr: np.ndarray, sl: np.ndarray) -> np.ndarray:
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def residual_ratio(y, x) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(y - mat @ coef) / den)


def main() -> int:
    ham0 = hop_H(N)
    c0, n0 = occupy(ham0, False)
    ham1 = ham0.copy()
    ham1[SRC, SRC] += EPS1
    ham2 = ham0.copy()
    ham2[SRC, SRC] += EPS2
    c1, n1 = occupy(ham1, False)
    c2, n2 = occupy(ham2, False)
    half = False
    if n1 != n0 or n2 != n0:
        half = True
        c0, n0 = occupy(ham0, True)
        c1, n1 = occupy(ham1, True)
        c2, n2 = occupy(ham2, True)
    de = site_energy(ham1, c1) - site_energy(ham0, c0)
    de_width = float(np.sqrt(np.average((np.arange(N) - SRC) ** 2, weights=np.abs(de) + 1e-18)))
    starts = list(range(0, N - L + 1))
    xs = np.arange(L) - (L - 1) / 2.0
    wchm = (L / 2.0) ** 2 - xs**2
    ds1, ds2, pchm, pflat = [], [], [], []
    pk0, pkmid = [], []
    dc = c1 - c0
    for s0 in starts:
        sl = np.arange(s0, s0 + L)
        ds1.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        ds2.append(peschel_s(c2, sl) - peschel_s(c0, sl))
        pchm.append(float(np.dot(wchm, de[sl])))
        pflat.append(float(np.sum(de[sl])))
        k0 = peschel_k(c0, sl)
        k1 = peschel_k(c1, sl)
        dcsl = dc[np.ix_(sl, sl)]
        pk0.append(float(np.sum(k0 * dcsl)))
        pkmid.append(float(np.sum(0.5 * (k0 + k1) * dcsl)))
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)
    pchm = np.array(pchm)
    pflat = np.array(pflat)
    pk0 = np.array(pk0)
    pkmid = np.array(pkmid)
    r_chm = residual_ratio(ds1, pchm)
    r_flat = residual_ratio(ds1, pflat)
    rho_c = pearson(ds1, pchm)
    rho_f = pearson(ds1, pflat)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6)
    c1g = bool(pearson(ds1, ds2) > 0.95)
    c2g = bool(r_chm < r_flat)
    c4g = bool(abs(rho_c) > 0.60)
    ok = bool(c0g and c1g and c2g and c4g)
    payload = {
        "task": "m9.27_1d_point_chm",
        "n_intervals": int(len(starts)),
        "half_fill": half,
        "n_occ": [int(n0), int(n1), int(n2)],
        "de_rms_width": de_width,
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "rho_eps": pearson(ds1, ds2),
        "rho_CHM": rho_c,
        "rho_flat": rho_f,
        "R_CHM": r_chm,
        "R_flat": r_flat,
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C2_PRIMARY_chm_beats_flat": c2g,
        "C4_tracks": c4g,
        "all_gates": ok,
        "rho_Kvac": pearson(ds1, pk0),
        "rho_Kmid": pearson(ds1, pkmid),
        "rel_err_Kmid": float(np.max(np.abs(ds1 - pkmid)) / np.max(np.abs(ds1))),
        "verdict": "1D_CHM_WINS" if ok else "1D_INSTRUMENT_OR_FLAT",
        "not_claimed": ["3d Einstein", "8pi G", "de Sitter"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_27_1d_point_chm.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
