#!/usr/bin/env python3
"""M9.23: Bloch-like dimer vs CHM on a complete ball covering.

The guess: virtual modes live on a Bloch 3-sphere. Testable
piece: the modular hop K_ij tracks the Bloch vector of the
2x2 correlator of that dimer better than the CHM weight.

Bloch-like coordinates of the real 2x2 block C of sites (i,j):
    r_x = 2 C_ij,   r_z = C_ii - C_jj,   r = hypot(r_x, r_z).
Primary Bloch observable: r_x (the hop / coherence).
This is Pauli coordinates of a correlator, not a qubit state.

DENSE COVERING: every spatial NN pair inside the ball.
Two balls: N=16 R=5 (515 sites) and N=12 R=4 (257 sites).
One eigh each. No hop perturbations. No subsample.

PRE-REGISTERED:
  C0  1d interval, every bond: |ρ(K, r_x)| > 0.50.
  C1  3d N=16 R=5, all NN: |ρ(K, r_x)| > 0.30 (not noise).
  C2  PRIMARY. R_shape(K, r_x) < R_shape(K, w_CHM)
      on N=16 R=5. Bloch beats CHM as a predictor of K.
  C3  R_shape(K, r_x) < R_shape(K, flat) on N=16 R=5.
  C4  Same C2 on N=12 R=4 (second covering).

r_z mean is reported. It is not identified with curvature.

Not claimed: Planck, eta=1/4G, Einstein, dS, Bloch habitat.

Writes ../data/m9_23_bloch.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N1, L1 = 240, 32


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
    if len(y) < 3:
        return float("nan")
    mat = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(y - mat @ coef) / den)


def score(k, bloch, wchm) -> dict:
    k = np.asarray(k, dtype=float)
    bloch = np.asarray(bloch, dtype=float)
    wchm = np.asarray(wchm, dtype=float)
    flat = np.ones_like(k)
    return {
        "n": int(k.size),
        "rho_bloch": pearson(k, bloch),
        "R_bloch": residual_ratio(k, bloch),
        "rho_chm": pearson(k, wchm),
        "R_chm": residual_ratio(k, wchm),
        "R_flat": residual_ratio(k, flat),
        "mean_abs_rx": float(np.mean(np.abs(bloch))),
        "std_rx": float(np.std(bloch)),
    }


def modular_and_C(ham, sl):
    ev, vecs = np.linalg.eigh(ham)
    corr = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    ca = corr[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(ca)
    w = np.clip(w, CLIP, 1.0 - CLIP)
    k = (u * np.log((1.0 - w) / w)) @ u.T
    return 0.5 * (k + k.T), ca


def one_d() -> dict:
    n, ell = N1, L1
    ham = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        ham[i, i + 1] = ham[i + 1, i] = -1.0
    mid = n // 2
    sl = np.arange(mid - ell // 2, mid + ell // 2)
    k, ca = modular_and_C(ham, sl)
    x = np.arange(ell) - (ell - 1) / 2.0
    knn, rx, wchm, rz = [], [], [], []
    for i in range(ell - 1):
        knn.append(k[i, i + 1])
        rx.append(2.0 * ca[i, i + 1])
        rz.append(ca[i, i] - ca[i + 1, i + 1])
        xm = 0.5 * (x[i] + x[i + 1])
        wchm.append((ell / 2.0) ** 2 - xm**2)
    out = score(knn, rx, wchm)
    out["mean_rz"] = float(np.mean(rz))
    out["C0"] = bool(abs(out["rho_bloch"]) > 0.50)
    return out


def idx(x, y, z, n) -> int:
    return (x * n + y) * n + z


def staggered_H(n: int) -> np.ndarray:
    vol = n**3
    ham = np.zeros((vol, vol), dtype=float)
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


def ball(n: int, radius: int):
    ham = staggered_H(n)
    c = n // 2
    sl, coords = [], []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius * radius:
                    sl.append(idx(x, y, z, n))
                    coords.append((x, y, z))
    sl = np.array(sl)
    k, ca = modular_and_C(ham, sl)
    pos = {p: i for i, p in enumerate(coords)}
    knn, rx, rz, wchm, rmag = [], [], [], [], []
    for i, (x, y, z) in enumerate(coords):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nbr = (x + d[0], y + d[1], z + d[2])
            j = pos.get(nbr)
            if j is None or j <= i:
                continue
            knn.append(k[i, j])
            rxi = 2.0 * ca[i, j]
            rzi = ca[i, i] - ca[j, j]
            rx.append(rxi)
            rz.append(rzi)
            rmag.append(float(np.hypot(rxi, rzi)))
            mx = 0.5 * (x + nbr[0] - 2 * c)
            my = 0.5 * (y + nbr[1] - 2 * c)
            mz = 0.5 * (z + nbr[2] - 2 * c)
            wchm.append(radius * radius - (mx * mx + my * my + mz * mz))
    out = score(knn, rx, wchm)
    out_mag = score(knn, rmag, wchm)
    out["R_bloch_mag"] = out_mag["R_bloch"]
    out["rho_bloch_mag"] = out_mag["rho_bloch"]
    out["mean_rz"] = float(np.mean(rz))
    out["N"] = n
    out["R"] = radius
    out["n_ball"] = int(sl.size)
    return out


def main() -> int:
    d1 = one_d()
    d_fine = ball(16, 5)
    d_coarse = ball(12, 4)
    c0 = bool(d1["C0"])
    c1 = bool(abs(d_fine["rho_bloch"]) > 0.30)
    c2 = bool(d_fine["R_bloch"] < d_fine["R_chm"])
    c3 = bool(d_fine["R_bloch"] < d_fine["R_flat"])
    c4 = bool(d_coarse["R_bloch"] < d_coarse["R_chm"])
    ok = bool(c0 and c1 and c2 and c3 and c4)
    payload = {
        "task": "m9.23_bloch",
        "covering": "ALL NN dimers in the ball; two radii",
        "one_d": d1,
        "ball_N16_R5": d_fine,
        "ball_N12_R4": d_coarse,
        "C0_instrument": c0,
        "C1_bloch_not_noise": c1,
        "C2_PRIMARY_bloch_beats_chm": c2,
        "C3_bloch_beats_flat": c3,
        "C4_second_covering": c4,
        "all_gates": ok,
        "verdict": (
            "BLOCH_BEATS_CHM"
            if ok
            else ("BLOCH_INSTRUMENT_REJECTED" if not c0 else "CHM_BEATS_BLOCH")
        ),
        "not_claimed": [
            "curvature axis",
            "Bloch habitat",
            "Planck",
            "eta=1/4G",
            "Einstein",
            "de Sitter",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_23_bloch.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
