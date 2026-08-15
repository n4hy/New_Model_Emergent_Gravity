#!/usr/bin/env python3
"""M9.28: 1d first law at fixed H, occupation transfer.

H never changes. A localized occupied packet L and its bipartite
stagger R (unoccupied) exchange occupation

    C(α) = C_0 + α (|R⟩⟨R| − |L⟩⟨L|).

Discarded (not scored):
  - Hamiltonian point source (Paper 36): δS ≠ Tr(K_vac ΔC).
  - Coherent particle-hole rotation: first-order ⟨H⟩ vanishes
    (bipartite selection) and C_vac is θ-independent and fails.
  - First-order dC from a potential: Tr(H_0 dC) = 0 (Hellmann).

PRE-REGISTERED:
  N=200, hop -1, L=16, 185 intervals, packet at site 100, σ=2.
  α=0.02 and α=0.05. H fixed. e_i = ∑_j H_ij C_ij.
  P_CHM = ∑_I ((L/2)^2 − x^2) δe
  P_flat = ∑_I δe
  C_vac  INSTRUMENT. |ρ(δS, Tr(K_vac ΔC))| > 0.95.
  C0     max|δS| > 1e-6.
  C1     Pearson(δS(α), δS(2.5α)) > 0.95.
  C2     PRIMARY, scored only if C_vac.
         R_shape(δS, P_CHM) < R_shape(δS, P_flat).
  C4     |ρ(δS, P_CHM)| > 0.60, scored only if C_vac.

Writes ../data/m9_28_1d_state.json
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
SIGMA = 2.0
ALPHA1 = 0.02
ALPHA2 = 0.05


def hop_H(n: int) -> np.ndarray:
    ham = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        ham[i, i + 1] = ham[i + 1, i] = -1.0
    return ham


def occupation_transfer(ham: np.ndarray, src: int, sigma: float, alpha: float):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    u_occ = vecs[:, occ]
    u_un = vecs[:, ~occ]
    xs = np.arange(ham.shape[0])
    env = np.exp(-0.5 * ((xs - src) / sigma) ** 2)
    stagger = 1.0 - 2.0 * (xs % 2)
    left = u_occ @ (u_occ.T @ env)
    right = u_un @ (u_un.T @ (stagger * env))
    n_l = float(np.linalg.norm(left))
    n_r = float(np.linalg.norm(right))
    if n_l < 1e-14 or n_r < 1e-14:
        raise RuntimeError("packet vanished")
    left = left / n_l
    right = right / n_r
    c0 = u_occ @ u_occ.T
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    corr = 0.5 * (corr + corr.T)
    return c0, corr, int(occ.sum()), n_l, n_r


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
    ham = hop_H(N)
    c0, c1, nocc, n_l, n_r = occupation_transfer(ham, SRC, SIGMA, ALPHA1)
    _, c2, _, _, _ = occupation_transfer(ham, SRC, SIGMA, ALPHA2)
    ev1 = np.linalg.eigvalsh(c1)
    de = site_energy(ham, c1) - site_energy(ham, c0)
    de_width = float(
        np.sqrt(np.average((np.arange(N) - SRC) ** 2, weights=np.abs(de) + 1e-18))
    )
    starts = list(range(0, N - L + 1))
    xs = np.arange(L) - (L - 1) / 2.0
    wchm = (L / 2.0) ** 2 - xs**2
    ds1, ds2, pchm, pflat, pk0 = [], [], [], [], []
    dc = c1 - c0
    for s0 in starts:
        sl = np.arange(s0, s0 + L)
        ds1.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        ds2.append(peschel_s(c2, sl) - peschel_s(c0, sl))
        pchm.append(float(np.dot(wchm, de[sl])))
        pflat.append(float(np.sum(de[sl])))
        pk0.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
    ds1 = np.asarray(ds1, dtype=float)
    ds2 = np.asarray(ds2, dtype=float)
    pchm = np.asarray(pchm, dtype=float)
    pflat = np.asarray(pflat, dtype=float)
    pk0 = np.asarray(pk0, dtype=float)
    r_chm = residual_ratio(ds1, pchm)
    r_flat = residual_ratio(ds1, pflat)
    rho_c = pearson(ds1, pchm)
    rho_k = pearson(ds1, pk0)
    c_vac = bool(abs(rho_k) > 0.95)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6)
    c1g = bool(pearson(ds1, ds2) > 0.95)
    c2g = bool(r_chm < r_flat)
    c4g = bool(abs(rho_c) > 0.60)
    ok = bool(c_vac and c0g and c1g and c2g and c4g)
    payload = {
        "task": "m9.28_1d_state",
        "H_fixed": True,
        "construction": "occupation_transfer",
        "sigma": SIGMA,
        "n_occ": int(nocc),
        "packet_norms": [n_l, n_r],
        "c_eig_min": float(np.min(ev1)),
        "c_eig_max": float(np.max(ev1)),
        "de_rms_width": de_width,
        "n_intervals": int(len(starts)),
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "rho_eps": pearson(ds1, ds2),
        "rho_CHM": rho_c,
        "rho_flat": pearson(ds1, pflat),
        "R_CHM": r_chm,
        "R_flat": r_flat,
        "rho_Kvac": rho_k,
        "C_vac": c_vac,
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C2_PRIMARY_chm_beats_flat": c2g,
        "C4_tracks": c4g,
        "all_gates": ok,
        "verdict": "1D_FIXEDH_CHM_WINS" if ok else (
            "1D_FIXEDH_INSTRUMENT_REJECT" if not c_vac else "1D_FIXEDH_FLAT"
        ),
        "not_claimed": ["3d Einstein", "8pi G", "de Sitter"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_28_1d_state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
