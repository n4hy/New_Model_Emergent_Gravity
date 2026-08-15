#!/usr/bin/env python3
"""M9.29: Paper 37 instrument on balls, cubes, taxicab regions.

The author's guess (NOT a result): the first-law kernel may need
a shape that is not a sphere. CHM is a ball theorem. This run
asks whether Paper 37's CHM win is sphere-specific.

Same state as M9.28: fixed H, occupation transfer.

PRE-REGISTERED:
  N=12. Packet (6,6,6), σ=1.5, α=0.02 and 0.05.
  Families (every legal center):
    balls R=2
    cubes side=3
    taxicab t=2
  Kernels, all ≥0 on the region:
    flat     1
    export   r_max² − r_eucl²   (CHM on a ball; illegal export off ball)
    native   ball: same as export
             cube: ∏_μ ((s/2)² − d_μ²)
             taxi: t² − ℓ1²
  Per family:
    C_vac  |ρ(δS, Tr(K_vac ΔC))| > 0.95
    C0     max|δS| > 1e-6
    C1     Pearson(δS(α), δS(2.5α)) > 0.95
    C2e    R(export) < R(flat)     scored only if C_vac
    C2n    R(native) < R(flat)     scored only if C_vac
    C2x    R(native) < R(export)   cubes/taxi only; the guess
  Control: balls must pass C_vac and C2e (Paper 37).
  PRIMARY: on cubes, C2e. If FAIL while balls PASS, Paper 37
  is sphere-specific. The guess is measured by cube C2n and C2x.
  Guess stays a guess even if C2x passes.

Writes ../data/m9_29_shape.json
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
SIGMA = 1.5
ALPHA1 = 0.02
ALPHA2 = 0.05


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H(n: int) -> np.ndarray:
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


def occupation_transfer(ham: np.ndarray, n: int, src, sigma: float, alpha: float):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    u_occ, u_un = vecs[:, occ], vecs[:, ~occ]
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
    left = u_occ @ (u_occ.T @ env)
    right = u_un @ (u_un.T @ stag)
    n_l, n_r = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if n_l < 1e-14 or n_r < 1e-14:
        raise RuntimeError("packet vanished")
    left, right = left / n_l, right / n_r
    c0 = u_occ @ u_occ.T
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return c0, 0.5 * (corr + corr.T), int(occ.sum())


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


def family_regions(kind: str, n: int):
    """Yield (sites, coords_rel) for every legal center."""
    if kind == "ball":
        rad, ext = 2, 2
        lo, hi = ext, n - ext
        for cx in range(lo, hi):
            for cy in range(lo, hi):
                for cz in range(lo, hi):
                    sites, rel = [], []
                    for x in range(n):
                        for y in range(n):
                            for z in range(n):
                                dx, dy, dz = x - cx, y - cy, z - cz
                                if dx * dx + dy * dy + dz * dz <= rad * rad:
                                    sites.append(idx(x, y, z, n))
                                    rel.append((dx, dy, dz))
                    yield np.array(sites, dtype=int), rel
    elif kind == "cube":
        half, side = 1, 3
        lo, hi = half, n - half
        for cx in range(lo, hi):
            for cy in range(lo, hi):
                for cz in range(lo, hi):
                    sites, rel = [], []
                    for dx in range(-half, half + 1):
                        for dy in range(-half, half + 1):
                            for dz in range(-half, half + 1):
                                sites.append(idx(cx + dx, cy + dy, cz + dz, n))
                                rel.append((dx, dy, dz))
                    yield np.array(sites, dtype=int), rel
                    _ = side
    elif kind == "taxi":
        t, ext = 2, 2
        lo, hi = ext, n - ext
        for cx in range(lo, hi):
            for cy in range(lo, hi):
                for cz in range(lo, hi):
                    sites, rel = [], []
                    for x in range(n):
                        for y in range(n):
                            for z in range(n):
                                dx, dy, dz = x - cx, y - cy, z - cz
                                if abs(dx) + abs(dy) + abs(dz) <= t:
                                    sites.append(idx(x, y, z, n))
                                    rel.append((dx, dy, dz))
                    yield np.array(sites, dtype=int), rel
    else:
        raise ValueError(kind)


def weights(kind: str, rel):
    r2 = np.array([dx * dx + dy * dy + dz * dz for dx, dy, dz in rel], dtype=float)
    export = float(np.max(r2)) - r2
    if kind == "ball":
        native = export.copy()
    elif kind == "cube":
        half2 = (3 / 2.0) ** 2
        native = np.array(
            [(half2 - dx * dx) * (half2 - dy * dy) * (half2 - dz * dz) for dx, dy, dz in rel],
            dtype=float,
        )
    else:
        native = np.array(
            [4.0 - (abs(dx) + abs(dy) + abs(dz)) ** 2 for dx, dy, dz in rel],
            dtype=float,
        )
    flat = np.ones(len(rel), dtype=float)
    return export, native, flat


def score_family(kind: str, ham, c0, c1, c2, de, dc) -> dict:
    ds1, ds2, pexp, pnat, pflat, pk0 = [], [], [], [], [], []
    n_sites = None
    for sl, rel in family_regions(kind, N):
        n_sites = int(len(sl))
        ds1.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        ds2.append(peschel_s(c2, sl) - peschel_s(c0, sl))
        we, wn, wf = weights(kind, rel)
        e = de[sl]
        pexp.append(float(np.dot(we, e)))
        pnat.append(float(np.dot(wn, e)))
        pflat.append(float(np.dot(wf, e)))
        pk0.append(float(np.sum(peschel_k(c0, sl) * dc[np.ix_(sl, sl)])))
    ds1 = np.asarray(ds1, dtype=float)
    ds2 = np.asarray(ds2, dtype=float)
    pexp = np.asarray(pexp, dtype=float)
    pnat = np.asarray(pnat, dtype=float)
    pflat = np.asarray(pflat, dtype=float)
    pk0 = np.asarray(pk0, dtype=float)
    r_e, r_n, r_f = residual_ratio(ds1, pexp), residual_ratio(ds1, pnat), residual_ratio(ds1, pflat)
    rho_k = pearson(ds1, pk0)
    c_vac = bool(abs(rho_k) > 0.95)
    c0g = bool(float(np.max(np.abs(ds1))) > 1e-6)
    c1g = bool(pearson(ds1, ds2) > 0.95)
    c2e = bool(r_e < r_f)
    c2n = bool(r_n < r_f)
    c2x = bool(r_n < r_e)
    return {
        "kind": kind,
        "n_regions": int(len(ds1)),
        "n_sites": n_sites,
        "max_abs_dS": float(np.max(np.abs(ds1))),
        "rho_eps": pearson(ds1, ds2),
        "rho_Kvac": rho_k,
        "rho_export": pearson(ds1, pexp),
        "rho_native": pearson(ds1, pnat),
        "rho_flat": pearson(ds1, pflat),
        "R_export": r_e,
        "R_native": r_n,
        "R_flat": r_f,
        "C_vac": c_vac,
        "C0_signal": c0g,
        "C1_linear": c1g,
        "C2e_export_beats_flat": c2e,
        "C2n_native_beats_flat": c2n,
        "C2x_native_beats_export": c2x,
        "scored": bool(c_vac),
    }


def main() -> int:
    ham = hop_H(N)
    c0, c1, nocc = occupation_transfer(ham, N, SRC, SIGMA, ALPHA1)
    _, c2, _ = occupation_transfer(ham, N, SRC, SIGMA, ALPHA2)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    dc = c1 - c0
    families = {k: score_family(k, ham, c0, c1, c2, de, dc) for k in ("ball", "cube", "taxi")}
    ball, cube, taxi = families["ball"], families["cube"], families["taxi"]
    control = bool(ball["C_vac"] and ball["C2e_export_beats_flat"])
    sphere_specific = bool(
        control and cube["C_vac"] and (not cube["C2e_export_beats_flat"]) and (not cube["C2n_native_beats_flat"])
    )
    guess_measured = bool(cube["C_vac"] and cube["C2n_native_beats_flat"] and cube["C2x_native_beats_export"])
    export_lives = bool(cube["C_vac"] and cube["C2e_export_beats_flat"])
    if not ball["C_vac"]:
        verdict = "INSTRUMENT_REJECT"
    elif sphere_specific:
        verdict = "SPHERE_SPECIFIC"
    elif guess_measured:
        verdict = "SHAPE_NATIVE_MEASURED_STILL_A_GUESS"
    elif export_lives:
        verdict = "CHM_EXPORTS_OFF_SPHERE"
    else:
        verdict = "SHAPE_MIXED"
    payload = {
        "task": "m9.29_shape",
        "H_fixed": True,
        "guess": "need a shape, not a sphere",
        "guess_status": "GUESS",
        "n_occ": int(nocc),
        "families": families,
        "control_ball_chm": control,
        "sphere_specific": sphere_specific,
        "guess_measured": guess_measured,
        "verdict": verdict,
        "not_claimed": ["Einstein equation", "8pi G", "de Sitter", "CHM off balls"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_29_shape.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if verdict in ("SPHERE_SPECIFIC", "SHAPE_NATIVE_MEASURED_STILL_A_GUESS", "CHM_EXPORTS_OFF_SPHERE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
