#!/usr/bin/env python3
"""M9.40 audit. N=10, A=(2,5,5), B=(6,5,5), σ=0.9, α=0.03.

Own κ from packet A, R=3. Pair ball at (4,5,5), R=4
(must enclose both packets, not just the two centres).
Own n=65 Poisson, SEP=0.20 L. Tries to REFUTE C_read and C_mid.

Writes ../data/m9_40_audit_pair.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SIG, ALPHA = 10, 0.9, 0.03
A, B, MID = (2, 5, 5), (6, 5, 5), (4, 5, 5)
R_CAL, R_PAIR = 3, 4
LBOX, NBOX, GCONST = 1.0, 65, 1.0
SEP = 0.20 * LBOX
EXTERIOR = (0.40 * LBOX, 0.45 * LBOX, 0.50 * LBOX)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def raw_packet(uo, uu, n, src, sigma):
    env = np.zeros(n**3)
    stag = np.zeros(n**3)
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
    return e1, v2 / np.linalg.norm(v2)


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
    xs = np.linspace(-LBOX, LBOX, NBOX)
    h = float(xs[1] - xs[0])
    m = NBOX - 2
    fhat = rhs[1:-1, 1:-1, 1:-1].copy()
    for ax in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=ax)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for ax in (0, 1, 2):
        phat = dst(phat, type=1, axis=ax)
    phi = np.zeros((NBOX, NBOX, NBOX))
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    return xs, h, phi


def interp(xs, field, p):
    def one(ax, val):
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


def deposit(de, mid, scale):
    xs = np.linspace(-LBOX, LBOX, NBOX)
    h = float(xs[1] - xs[0])
    rhs = np.zeros((NBOX, NBOX, NBOX))
    mx, my, mz = mid
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                px, py, pz = (x - mx) * scale, (y - my) * scale, (z - mz) * scale
                if max(abs(px), abs(py), abs(pz)) >= LBOX - 0.5 * h:
                    continue
                ix = int(np.argmin(np.abs(xs - px)))
                iy = int(np.argmin(np.abs(xs - py)))
                iz = int(np.argmin(np.abs(xs - pz)))
                if min(ix, iy, iz) <= 0 or max(ix, iy, iz) >= NBOX - 1:
                    continue
                rhs[ix, iy, iz] += 4.0 * np.pi * GCONST * de[i] / (h**3)
    return rhs


def main() -> int:
    vol = N**3
    ham = np.zeros((vol, vol))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        j = idx(xx, yy, zz)
                        ham[i, j] = ham[j, i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T
    la, ra = raw_packet(uo, uu, N, A, SIG)
    lb, rb = raw_packet(uo, uu, N, B, SIG)
    la_n = la / np.linalg.norm(la)
    ra_n = ra / np.linalg.norm(ra)
    ca = 0.5 * (
        (c0 + ALPHA * (np.outer(ra_n, ra_n) - np.outer(la_n, la_n)))
        + (c0 + ALPHA * (np.outer(ra_n, ra_n) - np.outer(la_n, la_n))).T
    )
    la, lb = orthonormalize(la, lb)
    ra, rb = orthonormalize(ra, rb)
    cab = c0 + ALPHA * (np.outer(ra, ra) - np.outer(la, la) + np.outer(rb, rb) - np.outer(lb, lb))
    cab = 0.5 * (cab + cab.T)
    de_a = np.sum(ham * ca, axis=1) - np.sum(ham * c0, axis=1)
    de_ab = np.sum(ham * cab, axis=1) - np.sum(ham * c0, axis=1)
    m_ab = float(de_ab.sum())
    sl_a = ball(A, R_CAL)
    kappa = (peschel_s(ca, sl_a) - peschel_s(c0, sl_a)) / float(np.sum(de_a[sl_a]))
    sl_p = ball(MID, R_PAIR)
    p_pair = float(np.sum(de_ab[sl_p]))
    m_fl = (peschel_s(cab, sl_p) - peschel_s(c0, sl_p)) / kappa
    read_rel = abs(m_fl / m_ab - 1.0)
    c_read = bool(read_rel < 0.10)

    scale = SEP / float(B[0] - A[0])
    rhs = deposit(de_ab, MID, scale)
    xs, h, phi = dst_poisson(rhs)
    ax = np.zeros_like(phi)
    ay = np.zeros_like(phi)
    az = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    ay[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * h)
    az[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * h)
    a_mid = np.array(
        [
            interp(xs, ax, (0.0, 0.0, 0.0)),
            interp(xs, ay, (0.0, 0.0, 0.0)),
            interp(xs, az, (0.0, 0.0, 0.0)),
        ]
    )
    a_char = GCONST * abs(m_ab) / (0.5 * SEP) ** 2
    mid_rel = float(np.linalg.norm(a_mid) / a_char)
    c_mid = bool(mid_rel < 0.10)

    pos_a = ((A[0] - MID[0]) * scale, 0.0, 0.0)
    pos_b = ((B[0] - MID[0]) * scale, 0.0, 0.0)
    # auditor Coulomb uses M_FL/2 each if we refuse sum-de split;
    # we use half of M_AB only if masses are equal — report M_FL vs M_AB.
    m_half = 0.5 * m_ab
    ext_res = []
    for r in EXTERIOR:
        p = (r, 0.0, 0.0)
        ap = np.array([interp(xs, ax, p), interp(xs, ay, p), interp(xs, az, p)])
        ac = np.zeros(3)
        for mass, pos in ((m_half, pos_a), (m_half, pos_b)):
            rvec = np.asarray(p) - np.asarray(pos)
            rr = float(np.linalg.norm(rvec))
            ac += -GCONST * mass * rvec / (rr**3)
        ext_res.append(float(np.linalg.norm(ap - ac) / np.linalg.norm(ac)))
    c_ext = bool(all(v < 0.10 for v in ext_res))

    payload = {
        "task": "m9.40_audit_pair",
        "kappa_cal": float(kappa),
        "M_AB": m_ab,
        "M_FL": float(m_fl),
        "P_pair": p_pair,
        "enclose_frac": p_pair / m_ab if m_ab else None,
        "read_rel": read_rel,
        "a_mid": a_mid.tolist(),
        "mid_rel": mid_rel,
        "exterior_res": ext_res,
        "C_read": c_read,
        "C_mid": c_mid,
        "C_ext": c_ext,
        "verdicts": {
            "C_read": "CONFIRMED" if c_read else "REFUTED",
            "C_mid": "CONFIRMED" if c_mid else "REFUTED",
            "C_ext": "CONFIRMED" if c_ext else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_40_audit_pair.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_read and c_mid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
