#!/usr/bin/env python3
"""M9.32: is area independent of energy, or the same bump twice?

Jacobson / FGHMV is two terms: δS = η δA + β δE.
Papers 39–41 have δS~δA with no constant η, and energy
beats area. This run asks whether δA carries anything
beyond P_CHM.

Two independent variations, same 512 balls:
  matter    occupation transfer, hops fixed  (δA ≡ 0)
  geometry  hop conformal bump, new GS

PRE-REGISTERED:
  N=12, R=2. Packet/Φ at (6,6,6), σ=1.5 / 2.0.
  α=0.02, ε=0.02. A = ∑_cut 1/t².
  C_vac     |ρ(δS, Tr(K_vac ΔC))| > 0.95 on each variation
  C_matter  |ρ(δS, P_CHM)| > 0.60 on matter; max|δA| < 1e-8
  C_colin   DIAGNOSTIC. |ρ(δA, P_CHM)| on geometry.
            |ρ| > 0.90 means not independent.
  C_partial |partial ρ(δS, δA | P)| > 0.50 on geometry.
            NOT scored if C_colin dependent: the partial is
            then a degeneracy artifact.
  C_joint   R(δS; [δA, P]) < R(δS; P) on stacked matter+geometry.
  PRIMARY   if C_colin dependent → AREA_IS_ENERGY_PROXY.

Not claimed: 8πG, Einstein, de Sitter, MODELS.md.

Writes ../data/m9_32_two_term.json
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
SIG_M = 1.5
SIG_G = 2.0
ALPHA = 0.02
EPS = 0.02


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H(n: int, phi=None, eps: float = 0.0) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        scale = 1.0
                        if phi is not None:
                            scale = 1.0 + eps * 0.5 * (phi[i] + phi[j])
                        ham[i, j] = ham[j, i] = -scale
    return ham


def phi_field(n, src, sigma):
    phi = np.zeros((n**3,), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                phi[idx(x, y, z, n)] = np.exp(-0.5 * rr / (sigma * sigma))
    return phi


def occupy(ham):
    ev, vecs = np.linalg.eigh(ham)
    fill = ev < 0.0
    return vecs[:, fill] @ vecs[:, fill].T, int(fill.sum())


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
    return c0, 0.5 * (corr + corr.T)


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def peschel_k(corr, sl):
    block = corr[np.ix_(sl, sl)]
    ev, vecs = np.linalg.eigh(block)
    ev = np.clip(ev, CLIP, 1.0 - CLIP)
    return (vecs * np.log((1.0 - ev) / ev)) @ vecs.T


def tmap(n, phi, eps):
    out = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        t = -(1.0 + eps * 0.5 * (phi[i] + phi[j]))
                        out[(min(i, j), max(i, j))] = t
    return out


def a_face(n, sl, tm):
    inside = set(int(i) for i in sl)
    acc = 0.0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                if i not in inside:
                    continue
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if not (0 <= xx < n and 0 <= yy < n and 0 <= zz < n):
                        acc += 1.0
                        continue
                    j = idx(xx, yy, zz, n)
                    if j not in inside:
                        t = tm[(min(i, j), max(i, j))]
                        acc += 1.0 / (t * t)
    return acc


def ball_sites(n, radius, cx, cy, cz):
    r2 = radius * radius
    return np.array(
        [
            idx(x, y, z, n)
            for x in range(n)
            for y in range(n)
            for z in range(n)
            if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2
        ],
        dtype=int,
    )


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def residual_ratio(y, *cols):
    y = np.asarray(y, float)
    mat = np.column_stack(list(cols) + [np.ones(len(y))])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    den = float(np.linalg.norm(y - y.mean()))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(y - mat @ coef) / den), coef


def partial_rho(s, a, p):
    rsa, rsp, rap = pearson(s, a), pearson(s, p), pearson(a, p)
    den = np.sqrt((1.0 - rsp * rsp) * (1.0 - rap * rap))
    if den == 0.0 or not np.isfinite(den):
        return float("nan")
    return float((rsa - rsp * rap) / den)


def main() -> int:
    ham0 = hop_H(N)
    c0, n0 = occupy(ham0)
    # matter
    _, cm = occupation_transfer(ham0, N, SRC, SIG_M, ALPHA)
    # geometry
    phi = phi_field(N, SRC, SIG_G)
    hamg = hop_H(N, phi, EPS)
    cg, ng = occupy(hamg)
    tm0 = tmap(N, phi, 0.0)
    tmg = tmap(N, phi, EPS)
    dem = np.sum(ham0 * cm, axis=1) - np.sum(ham0 * c0, axis=1)
    deg = np.sum(hamg * cg, axis=1) - np.sum(ham0 * c0, axis=1)
    dcm, dcg = cm - c0, cg - c0
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    sm, am, pm, km = [], [], [], []
    sg, ag, pg, kg = [], [], [], []
    for cx, cy, cz in centers:
        sl = ball_sites(N, RADIUS, cx, cy, cz)
        sm.append(peschel_s(cm, sl) - peschel_s(c0, sl))
        sg.append(peschel_s(cg, sl) - peschel_s(c0, sl))
        am.append(a_face(N, sl, tm0) - a_face(N, sl, tm0))
        ag.append(a_face(N, sl, tmg) - a_face(N, sl, tm0))
        wchm = []
        em = eg = 0.0
        ecm = ecg = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    rr = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if rr <= r2max:
                        ww = r2max - rr
                        em += ww * dem[idx(x, y, z)]
                        eg += ww * deg[idx(x, y, z)]
                        ecm += dem[idx(x, y, z)]
                        ecg += deg[idx(x, y, z)]
        pm.append(em)
        pg.append(eg)
        km.append(float(np.sum(peschel_k(c0, sl) * dcm[np.ix_(sl, sl)])))
        kg.append(float(np.sum(peschel_k(c0, sl) * dcg[np.ix_(sl, sl)])))
        _ = (ecm, ecg, wchm)
    sm, am, pm = map(np.asarray, (sm, am, pm))
    sg, ag, pg = map(np.asarray, (sg, ag, pg))
    km, kg = np.asarray(km), np.asarray(kg)
    rho_colin = pearson(ag, pg)
    part = partial_rho(sg, ag, pg)
    r_p_g, _ = residual_ratio(sg, pg)
    r_a_g, _ = residual_ratio(sg, ag)
    r_both_g, coef_g = residual_ratio(sg, ag, pg)
    # stacked matter + geometry
    ys = np.concatenate([sm, sg])
    ya = np.concatenate([am, ag])
    yp = np.concatenate([pm, pg])
    r_p_s, _ = residual_ratio(ys, yp)
    r_both_s, coef_s = residual_ratio(ys, ya, yp)
    c_vac_m = bool(abs(pearson(sm, km)) > 0.95)
    c_vac_g = bool(abs(pearson(sg, kg)) > 0.95)
    c_matter = bool(abs(pearson(sm, pm)) > 0.60 and float(np.max(np.abs(am))) < 1e-8)
    c_colin_dep = bool(abs(rho_colin) > 0.90)
    c_partial = bool(np.isfinite(part) and abs(part) > 0.50)
    c_joint = bool(r_both_s < r_p_s)
    if not (c_vac_m and c_vac_g):
        verdict = "INSTRUMENT_REJECT"
    elif c_colin_dep:
        verdict = "AREA_IS_ENERGY_PROXY"
    elif c_partial and c_joint:
        verdict = "TWO_TERM_INDEPENDENT"
    elif c_partial:
        verdict = "PARTIAL_ONLY"
    else:
        verdict = "ENERGY_SUFFICES"
    payload = {
        "task": "m9.32_two_term",
        "n_balls": int(len(centers)),
        "n_occ": [int(n0), int(ng)],
        "matter": {
            "rho_Kvac": pearson(sm, km),
            "rho_CHM": pearson(sm, pm),
            "max_abs_dA": float(np.max(np.abs(am))),
            "max_abs_dS": float(np.max(np.abs(sm))),
        },
        "geometry": {
            "rho_Kvac": pearson(sg, kg),
            "rho_CHM": pearson(sg, pg),
            "rho_area": pearson(sg, ag),
            "rho_A_vs_P": rho_colin,
            "partial_rho_S_A_given_P": part,
            "R_energy": r_p_g,
            "R_area": r_a_g,
            "R_both": r_both_g,
            "coef_eta_beta_intercept": [float(x) for x in coef_g],
        },
        "stacked": {
            "R_energy": r_p_s,
            "R_both": r_both_s,
            "coef_eta_beta_intercept": [float(x) for x in coef_s],
        },
        "C_vac_matter": c_vac_m,
        "C_vac_geometry": c_vac_g,
        "C_matter": c_matter,
        "C_colin_dependent": c_colin_dep,
        "C_partial_raw": c_partial,
        "C_partial_scored": bool(c_partial and not c_colin_dep),
        "C_joint": c_joint,
        "verdict": verdict,
        "not_claimed": ["8pi G", "Einstein equation", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_32_two_term.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if verdict in ("TWO_TERM_INDEPENDENT", "AREA_IS_ENERGY_PROXY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
