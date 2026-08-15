#!/usr/bin/env python3
"""M9.41: wide packet — volume first law, Newtonian Λ (a ∝ r).

Compact packets (Papers 47–50) are stars: δS plateaus, a ~ 1/r².
A nearly uniform energy density is the Newtonian cosmological
constant: enclosed mass grows with volume, a ∝ r inside.

PRE-REGISTERED:
  N=12, source (6,6,6), α=0.02. Wide: σ=8. Compact control: σ=1.
  Source-centered balls R=2,3,4,5.
  V = site count. A = number of NN bonds leaving the ball.
  C_fl / C_grow PRIMARY (no Poisson). Nested V vs A is
      collinear — not a gate (Paper 42 degeneracy).
      Pearson(δS, P_flat) > 0.95
      and δS(R=5)/δS(R=2) > 1.30  (does not plateau)
  Compact control: δS(5)/δS(2) < 1.15 (must plateau).
  C_lin inherited Poisson of the wide δe at r=0.10,0.15,0.20 L:
      recorded, not required. Wide occupation transfer is
      not a uniform fluid (std/mean diagnostic).

Not claimed: derived Poisson, 8πG, FGHMV, de Sitter dual,
MODELS.md. a ∝ r is the Newtonian Λ signature of uniform ρ.

Writes ../data/m9_41_uniform.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
SRC = (6, 6, 6)
ALPHA = 0.02
SIG_WIDE = 8.0
SIG_COMP = 1.0
RADII = (2, 3, 4, 5)
LBOX = 1.0
NBOX = 65
GCONST = 1.0
PROBES = (0.10 * LBOX, 0.15 * LBOX, 0.20 * LBOX)


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


def occupation_transfer(uo, uu, n, src, sigma, alpha):
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


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def ball_and_area(center, radius, n=N):
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
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < 0 or yy < 0 or zz < 0 or xx >= n or yy >= n or zz >= n:
                        area += 1
                    elif not inside[xx, yy, zz]:
                        area += 1
    return np.array(sl, dtype=int), int(np.sum(inside)), area


def dst_poisson(rhs):
    nbox = rhs.shape[0]
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    m = nbox - 2
    fhat = rhs[1:-1, 1:-1, 1:-1].copy()
    for ax in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=ax)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for ax in (0, 1, 2):
        phat = dst(phat, type=1, axis=ax)
    phi = np.zeros((nbox, nbox, nbox), dtype=float)
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    return xs, h, phi


def interp(xs, field, p):
    def one(ax, val):
        if val <= ax[0]:
            return 0, 0.0
        if val >= ax[-1]:
            return len(ax) - 2, 1.0
        k = int(np.searchsorted(ax, val) - 1)
        k = max(0, min(k, len(ax) - 2))
        return k, (val - ax[k]) / (ax[k + 1] - ax[k])

    i, tx = one(xs, p[0])
    j, ty = one(xs, p[1])
    k, tz = one(xs, p[2])
    acc = 0.0
    for di, wi in ((0, 1 - tx), (1, tx)):
        for dj, wj in ((0, 1 - ty), (1, ty)):
            for dk, wk in ((0, 1 - tz), (1, tz)):
                acc += wi * wj * wk * field[i + di, j + dj, k + dk]
    return float(acc)


def deposit_fill(de, n, src, nbox):
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    scale = 2.0 * LBOX / (n - 1)
    rhs = np.zeros((nbox, nbox, nbox), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                px, py, pz = (x - sx) * scale, (y - sy) * scale, (z - sz) * scale
                if max(abs(px), abs(py), abs(pz)) >= LBOX - 0.5 * h:
                    continue
                ix = int(np.argmin(np.abs(xs - px)))
                iy = int(np.argmin(np.abs(xs - py)))
                iz = int(np.argmin(np.abs(xs - pz)))
                if min(ix, iy, iz) <= 0 or max(ix, iy, iz) >= nbox - 1:
                    continue
                rhs[ix, iy, iz] += 4.0 * np.pi * GCONST * de[i] / (h**3)
    return rhs


def radii_scan(c0, c1, de, src):
    ds, pflat, vol, area = [], [], [], []
    for rad in RADII:
        sl, v, a = ball_and_area(src, rad)
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pflat.append(float(np.sum(de[sl])))
        vol.append(v)
        area.append(a)
    return map(np.asarray, (ds, pflat, vol, area))


def main() -> int:
    ham = hop_H(N)
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0, c_w = occupation_transfer(uo, uu, N, SRC, SIG_WIDE, ALPHA)
    _, c_c = occupation_transfer(uo, uu, N, SRC, SIG_COMP, ALPHA)
    de_w = np.sum(ham * c_w, axis=1) - np.sum(ham * c0, axis=1)
    de_c = np.sum(ham * c_c, axis=1) - np.sum(ham * c0, axis=1)
    mean_abs = float(np.mean(np.abs(de_w)))
    unif = float(np.std(de_w) / mean_abs) if mean_abs else None

    ds_w, p_w, v_w, a_w = radii_scan(c0, c_w, de_w, SRC)
    ds_c, p_c, v_c, a_c = radii_scan(c0, c_c, de_c, SRC)
    rho_p = pearson(ds_w, p_w)
    rho_v = pearson(ds_w, v_w)
    rho_a = pearson(ds_w, a_w)
    grow_w = float(ds_w[-1] / ds_w[0]) if ds_w[0] else None
    grow_c = float(ds_c[-1] / ds_c[0]) if ds_c[0] else None
    c_vol = bool(abs(rho_p) > 0.95 and grow_w is not None and grow_w > 1.30)
    c_comp = bool(grow_c is not None and grow_c < 1.15)
    c_va = bool(abs(rho_v) > abs(rho_a))  # diagnostic; nested balls collinear

    rhs = deposit_fill(de_w, N, SRC, NBOX)
    xs, h, phi = dst_poisson(rhs)
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    accs = [interp(xs, ax, (r, 0.0, 0.0)) for r in PROBES]
    lr = np.log(np.asarray(PROBES, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    slope = float(
        np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)[0][0]
    )
    m_glob = float(np.sum(de_w))
    attractive = all(a < 0.0 for a in accs) if m_glob > 0.0 else all(a > 0.0 for a in accs)
    c_lin = bool(abs(slope - 1.0) < 0.40 and abs(slope - 1.0) < abs(slope + 2.0) and attractive)
    c_invsq_fail = bool(abs(slope + 2.0) > 0.50)
    ok = bool(c_vol and c_comp)
    payload = {
        "task": "m9.41_uniform",
        "uniformity_std_over_meanabs": unif,
        "M_wide": m_glob,
        "M_comp": float(np.sum(de_c)),
        "wide": {
            "deltaS": ds_w.tolist(),
            "P_flat": p_w.tolist(),
            "V": v_w.tolist(),
            "A": a_w.tolist(),
            "rho_P": rho_p,
            "rho_V": rho_v,
            "rho_A": rho_a,
            "grow": grow_w,
        },
        "compact": {
            "deltaS": ds_c.tolist(),
            "P_flat": p_c.tolist(),
            "grow": grow_c,
        },
        "a_r": accs,
        "slope": slope,
        "C_fl_grow_PRIMARY": c_vol,
        "C_VA_diagnostic": c_va,
        "C_compact_control": c_comp,
        "C_lin_recorded": c_lin,
        "C_invsq_FAIL": c_invsq_fail,
        "all_gates": ok,
        "verdict": "EXTENDED_FIRST_LAW" if ok else "UNIFORM_FAIL",
        "not_claimed": [
            "derived Poisson",
            "8pi G",
            "FGHMV",
            "de Sitter dual",
            "MODELS.md",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_41_uniform.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
