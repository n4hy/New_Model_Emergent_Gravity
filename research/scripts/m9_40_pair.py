#!/usr/bin/env python3
"""M9.40: two real packets. No M_hat, no point mass, no ball cloud.

Fewer assumptions than Papers 48–49:
  mass from one source-centered first-law ball, not a 60-ball median;
  ρ is the actual δe of the pair, not a point of M_hat;
  no 3-site compactification-as-physics (scale only sets the pair
  separation on the locked Poisson cube).

Still assumed: inherited DST Poisson, G=1. Not a derivation.

PRE-REGISTERED:
  N=12, packets A=(3,6,6), B=(9,6,6), σ=1, α=0.02, orthonormal
  two-source C as M9.34/35.
  κ_cal from ONE packet at A, source-centered R=3:
      κ = δS_A / P_flat_A
  Pair ball: centre (6,6,6), R=5. (R=4 encloses only 82% of
  ∑δe; Gauss requires the ball to hold the packets, not just
  the two site-centres.)
      M_FL = δS_AB / κ
      M_AB = ∑ δe_AB
  Embed on n=65, L=1, G=1. Midpoint → origin.
  6 sites of separation → SEP=0.20 L.
  C_eig   correlation eigs in [0, 1] to 1e-9
  C_add   |M_AB − M_A − M_B| / |M_AB| < 0.08
  C_read  PRIMARY. |M_FL / M_AB − 1| < 0.10
  C_mid   PRIMARY. |a_AB(0)| / a_char < 0.10
          a_char = G M_AB / (SEP/2)²
  C_ext   Poisson a vs two-point Coulomb of (M_A, pos_A)
          and (M_B, pos_B) at x=0.40L, 0.45L, 0.50L.
          residual < 0.10 at all three, a_x < 0
  C_super SANITY (linear Poisson). Not a discovery.
          rms(a_AB − a_A − a_B) / rms(a_AB) < 0.02
          at the three exterior probes plus the origin.

Not claimed: derived Poisson, 8πG, 1/4G, FGHMV, dS, MODELS.md.

Writes ../data/m9_40_pair.json
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
A = (3, 6, 6)
B = (9, 6, 6)
MID = (6, 6, 6)
SIGMA = 1.0
ALPHA = 0.02
R_CAL = 3
R_PAIR = 5
LBOX = 1.0
NBOX = 65
GCONST = 1.0
SEP = 0.20 * LBOX
EXTERIOR = (0.40 * LBOX, 0.45 * LBOX, 0.50 * LBOX)


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
    return uo @ (uo.T @ env), uu @ (uu.T @ stag)


def orthonormalize(v1, v2):
    e1 = v1 / np.linalg.norm(v1)
    v2 = v2 - e1 * np.dot(e1, v2)
    n2 = np.linalg.norm(v2)
    if n2 < 1e-14:
        raise RuntimeError("packets linearly dependent")
    return e1, v2 / n2


def one_source(uo, uu, c0, n, src, sigma, alpha):
    left, right = raw_packet(uo, uu, n, src, sigma)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return 0.5 * (corr + corr.T)


def two_source(uo, uu, c0, n, src_a, src_b, sigma, alpha):
    la, ra = raw_packet(uo, uu, n, src_a, sigma)
    lb, rb = raw_packet(uo, uu, n, src_b, sigma)
    la, lb = orthonormalize(la, lb)
    ra, rb = orthonormalize(ra, rb)
    corr = (
        c0
        + alpha * (np.outer(ra, ra) - np.outer(la, la))
        + alpha * (np.outer(rb, rb) - np.outer(lb, lb))
    )
    return 0.5 * (corr + corr.T)


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def ball(center, radius, n=N):
    cx, cy, cz = center
    return np.array(
        [
            idx(x, y, z, n)
            for x in range(n)
            for y in range(n)
            for z in range(n)
            if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius
        ],
        dtype=int,
    )


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


def deposit(de, n, mid, scale, nbox):
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    rhs = np.zeros((nbox, nbox, nbox), dtype=float)
    mx, my, mz = mid
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                px = (x - mx) * scale
                py = (y - my) * scale
                pz = (z - mz) * scale
                if max(abs(px), abs(py), abs(pz)) >= LBOX - 0.5 * h:
                    continue
                ix = int(np.argmin(np.abs(xs - px)))
                iy = int(np.argmin(np.abs(xs - py)))
                iz = int(np.argmin(np.abs(xs - pz)))
                if min(ix, iy, iz) <= 0 or max(ix, iy, iz) >= nbox - 1:
                    continue
                rhs[ix, iy, iz] += 4.0 * np.pi * GCONST * de[i] / (h**3)
    return rhs


def accel_field(rhs):
    xs, h, phi = dst_poisson(rhs)
    ax = np.zeros_like(phi)
    ay = np.zeros_like(phi)
    az = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    ay[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * h)
    az[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * h)
    return xs, ax, ay, az


def coulomb(p, masses, positions):
    acc = np.zeros(3, dtype=float)
    for mass, pos in zip(masses, positions):
        rvec = np.asarray(p, float) - np.asarray(pos, float)
        r = float(np.linalg.norm(rvec))
        if r < 1e-14:
            continue
        acc += -GCONST * mass * rvec / (r**3)
    return acc


def main() -> int:
    ham = hop_H(N)
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T
    ca = one_source(uo, uu, c0, N, A, SIGMA, ALPHA)
    cb = one_source(uo, uu, c0, N, B, SIGMA, ALPHA)
    cab = two_source(uo, uu, c0, N, A, B, SIGMA, ALPHA)
    eigs = np.linalg.eigvalsh(cab)
    c_eig = bool(eigs.min() >= -1e-9 and eigs.max() <= 1.0 + 1e-9)
    de_a = np.sum(ham * ca, axis=1) - np.sum(ham * c0, axis=1)
    de_b = np.sum(ham * cb, axis=1) - np.sum(ham * c0, axis=1)
    de_ab = np.sum(ham * cab, axis=1) - np.sum(ham * c0, axis=1)
    m_a, m_b, m_ab = float(de_a.sum()), float(de_b.sum()), float(de_ab.sum())
    add_rel = abs(m_ab - m_a - m_b) / abs(m_ab) if m_ab else None
    c_add = bool(add_rel is not None and add_rel < 0.08)

    sl_a = ball(A, R_CAL)
    kappa = (peschel_s(ca, sl_a) - peschel_s(c0, sl_a)) / float(np.sum(de_a[sl_a]))
    sl_p = ball(MID, R_PAIR)
    p_pair = float(np.sum(de_ab[sl_p]))
    ds_ab = peschel_s(cab, sl_p) - peschel_s(c0, sl_p)
    m_fl = ds_ab / kappa
    read_rel = abs(m_fl / m_ab - 1.0) if m_ab else None
    c_read = bool(read_rel is not None and read_rel < 0.10)

    site_sep = float(B[0] - A[0])
    scale = SEP / site_sep
    xs, ax_ab, ay_ab, az_ab = accel_field(deposit(de_ab, N, MID, scale, NBOX))
    _, ax_a, ay_a, az_a = accel_field(deposit(de_a, N, MID, scale, NBOX))
    _, ax_b, ay_b, az_b = accel_field(deposit(de_b, N, MID, scale, NBOX))

    a_mid = np.array(
        [
            interp(xs, ax_ab, (0.0, 0.0, 0.0)),
            interp(xs, ay_ab, (0.0, 0.0, 0.0)),
            interp(xs, az_ab, (0.0, 0.0, 0.0)),
        ]
    )
    a_char = GCONST * abs(m_ab) / (0.5 * SEP) ** 2
    mid_rel = float(np.linalg.norm(a_mid) / a_char) if a_char else None
    c_mid = bool(mid_rel is not None and mid_rel < 0.10)

    pos_a = ((A[0] - MID[0]) * scale, 0.0, 0.0)
    pos_b = ((B[0] - MID[0]) * scale, 0.0, 0.0)
    ext_poiss, ext_coul, ext_res = [], [], []
    for r in EXTERIOR:
        p = (r, 0.0, 0.0)
        ap = np.array(
            [interp(xs, ax_ab, p), interp(xs, ay_ab, p), interp(xs, az_ab, p)]
        )
        ac = coulomb(p, (m_a, m_b), (pos_a, pos_b))
        ext_poiss.append(ap.tolist())
        ext_coul.append(ac.tolist())
        denom = float(np.linalg.norm(ac))
        ext_res.append(float(np.linalg.norm(ap - ac) / denom) if denom else None)
    c_ext = bool(
        all(v is not None and v < 0.10 for v in ext_res)
        and all(row[0] < 0.0 for row in ext_poiss)
    )

    probes = [(0.0, 0.0, 0.0)] + [(r, 0.0, 0.0) for r in EXTERIOR]
    num = 0.0
    den = 0.0
    for p in probes:
        a12 = np.array(
            [interp(xs, ax_ab, p), interp(xs, ay_ab, p), interp(xs, az_ab, p)]
        )
        a1 = np.array([interp(xs, ax_a, p), interp(xs, ay_a, p), interp(xs, az_a, p)])
        a2 = np.array([interp(xs, ax_b, p), interp(xs, ay_b, p), interp(xs, az_b, p)])
        num += float(np.sum((a12 - a1 - a2) ** 2))
        den += float(np.sum(a12**2))
    super_rms = float(np.sqrt(num / den)) if den else None
    c_super = bool(super_rms is not None and super_rms < 0.02)

    ok = bool(c_eig and c_add and c_read and c_mid and c_ext)
    payload = {
        "task": "m9.40_pair",
        "kappa_cal": float(kappa),
        "M_A": m_a,
        "M_B": m_b,
        "M_AB": m_ab,
        "M_FL": float(m_fl),
        "P_pair": p_pair,
        "enclose_frac": (p_pair / m_ab) if m_ab else None,
        "add_rel": add_rel,
        "read_rel": read_rel,
        "a_mid": a_mid.tolist(),
        "mid_rel": mid_rel,
        "a_char": a_char,
        "exterior_poisson": ext_poiss,
        "exterior_coulomb": ext_coul,
        "exterior_res": ext_res,
        "super_rms": super_rms,
        "eig_min": float(eigs.min()),
        "eig_max": float(eigs.max()),
        "scale": scale,
        "SEP": SEP,
        "C_eig": c_eig,
        "C_add": c_add,
        "C_read_PRIMARY": c_read,
        "C_mid_PRIMARY": c_mid,
        "C_ext": c_ext,
        "C_super_SANITY": c_super,
        "all_gates": ok,
        "verdict": "PAIR_NEWTON" if ok else "PAIR_FAIL",
        "not_claimed": [
            "derived Poisson",
            "8pi G",
            "1/4G",
            "FGHMV",
            "de Sitter",
            "MODELS.md",
            "superposition is a discovery",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_40_pair.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
