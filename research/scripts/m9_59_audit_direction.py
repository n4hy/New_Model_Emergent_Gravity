#!/usr/bin/env python3
"""M9.59 audit. N=10. Own pair. mpmath dps=50. Tries to REFUTE C_dir.

A=(2,5,5), B=(7,5,5), σ=0.9, α_A=0.015, α_B=0.045.
R=3. Same n = lattice ∇_c M_AB vs Newton / CM.

Writes ../data/m9_59_audit_direction.json
"""

from __future__ import annotations

import json
import os

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
mp.mp.dps = 50
N = 10
SRC_A = (2, 5, 5)
SRC_B = (7, 5, 5)
SIG = 0.9
ALPHA_A = 0.015
ALPHA_B = 0.045
RADIUS = 3
FRAC_LO = mp.mpf("0.15")
FRAC_HI = mp.mpf("0.92")
GRAD_FRAC = mp.mpf("1e-6")


def mpf(x):
    return mp.mpf(x) if not isinstance(x, mp.mpf) else x


def fnum(x):
    if x is None:
        return None
    return float(x)


def mp_median(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


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


def angle_deg(u, v):
    nu = mp.sqrt(sum(x * x for x in u))
    nv = mp.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return None
    c = sum(x * y for x, y in zip(u, v)) / (nu * nv)
    if c > 1:
        c = mp.mpf(1)
    if c < -1:
        c = mp.mpf(-1)
    return mp.degrees(mp.acos(c))


def newton_g(c, pos_a, pos_b, m_a, m_b):
    g = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    for pos, mass in ((pos_a, m_a), (pos_b, m_b)):
        d = [pos[i] - mpf(c[i]) for i in range(3)]
        r2 = sum(x * x for x in d)
        if r2 == 0:
            continue
        r = mp.sqrt(r2)
        fac = mass / (r2 * r)
        for i in range(3):
            g[i] += fac * d[i]
    return g


def invsq_null_x(x_a, x_b, m_a, m_b):
    length = mpf(x_b) - mpf(x_a)
    ratio = mp.sqrt(m_b / m_a)
    return mpf(x_a) + length / (1 + ratio)


def main() -> int:
    ham = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < N and yy < N and zz < N:
                        ham[i, idx(xx, yy, zz)] = ham[idx(xx, yy, zz), i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T
    e0 = np.sum(ham * c0, axis=1)

    def one_de(src, alpha):
        left, right = raw_packet(uo, uu, N, src, SIG)
        left /= np.linalg.norm(left)
        right /= np.linalg.norm(right)
        corr = 0.5 * (
            (c0 + alpha * (np.outer(right, right) - np.outer(left, left)))
            + (c0 + alpha * (np.outer(right, right) - np.outer(left, left))).T
        )
        return [mpf(x) for x in (np.sum(ham * corr, axis=1) - e0)]

    def pair_de(aa, ab):
        la, ra = raw_packet(uo, uu, N, SRC_A, SIG)
        lb, rb = raw_packet(uo, uu, N, SRC_B, SIG)
        la, lb = orthonormalize(la, lb)
        ra, rb = orthonormalize(ra, rb)
        corr = (
            c0
            + aa * (np.outer(ra, ra) - np.outer(la, la))
            + ab * (np.outer(rb, rb) - np.outer(lb, lb))
        )
        corr = 0.5 * (corr + corr.T)
        return [mpf(x) for x in (np.sum(ham * corr, axis=1) - e0)]

    de_a = one_de(SRC_A, ALPHA_A)
    de_b = one_de(SRC_B, ALPHA_B)
    de_ab = pair_de(ALPHA_A, ALPHA_B)
    m_a = sum(de_a, mp.mpf(0))
    m_b = sum(de_b, mp.mpf(0))
    m_tot = sum(de_ab, mp.mpf(0))
    add_rel = abs(m_tot - m_a - m_b) / abs(m_tot)
    pos_a = [mpf(x) for x in SRC_A]
    pos_b = [mpf(x) for x in SRC_B]
    r_cm = [(m_a * pos_a[i] + m_b * pos_b[i]) / (m_a + m_b) for i in range(3)]
    lo, hi = RADIUS, N - RADIUS
    centres = [
        (x, y, z)
        for x in range(lo, hi)
        for y in range(lo, hi)
        for z in range(lo, hi)
    ]
    masses = {}
    for c in centres:
        sl = [
            idx(x, y, z)
            for x in range(N)
            for y in range(N)
            for z in range(N)
            if (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= RADIUS * RADIUS
        ]
        acc = mp.mpf(0)
        for i in sl:
            acc += de_ab[i]
        masses[c] = acc

    def grad_at(c):
        g = []
        for ax in range(3):
            cp, cm = list(c), list(c)
            cp[ax] += 1
            cm[ax] -= 1
            tp, tm = tuple(cp), tuple(cm)
            if tp not in masses or tm not in masses:
                return None
            g.append((masses[tp] - masses[tm]) / 2)
        return g

    floor = GRAD_FRAC * abs(m_tot)
    angs_n, angs_cm = [], []
    for c, mass in masses.items():
        gm = grad_at(c)
        if gm is None:
            continue
        gn = mp.sqrt(sum(x * x for x in gm))
        frac = mass / m_tot if m_tot != 0 else mp.mpf(0)
        if gn <= floor or frac <= FRAC_LO or frac >= FRAC_HI:
            continue
        ang_n = angle_deg(gm, newton_g(c, pos_a, pos_b, m_a, m_b))
        ang_cm = angle_deg(gm, [r_cm[i] - mpf(c[i]) for i in range(3)])
        if ang_n is None:
            continue
        angs_n.append(ang_n)
        if ang_cm is not None:
            angs_cm.append(ang_cm)
    med_n = mp_median(angs_n)
    med_cm = mp_median(angs_cm)
    null_x = invsq_null_x(pos_a[0], pos_b[0], m_a, m_b)
    axis = [
        c
        for c in masses
        if c[1] == SRC_A[1] and c[2] == SRC_A[2] and grad_at(c) is not None
    ]
    c_null = min(axis, key=lambda c: abs(mpf(c[0]) - null_x)) if axis else None
    c_cm = min(axis, key=lambda c: abs(mpf(c[0]) - r_cm[0])) if axis else None
    g_null = (
        mp.sqrt(sum(x * x for x in grad_at(c_null))) if c_null else None
    )
    g_cm = mp.sqrt(sum(x * x for x in grad_at(c_cm))) if c_cm else None
    c_dir = bool(len(angs_n) >= 4 and med_n is not None and med_n < 20)
    c_notcm = bool(
        len(angs_n) >= 4 and med_n is not None and med_cm is not None and med_n < med_cm
    )
    c_nullg = bool(g_null is not None and g_cm is not None and g_null < g_cm)
    payload = {
        "task": "m9.59_audit_direction",
        "precision": {
            "eigh": "float64 LAPACK",
            "analysis": "mpmath dps=50",
            "additivity_rel": fnum(add_rel),
        },
        "M_A": fnum(m_a),
        "M_B": fnum(m_b),
        "M_AB": fnum(m_tot),
        "n_keep": len(angs_n),
        "median_ang_newton": fnum(med_n),
        "median_ang_cm": fnum(med_cm),
        "null_x": fnum(null_x),
        "cm_x": fnum(r_cm[0]),
        "grad_at_null": fnum(g_null),
        "grad_at_cm": fnum(g_cm),
        "C_dir_PRIMARY": c_dir,
        "C_notcm": c_notcm,
        "C_null": c_nullg,
        "verdicts": {
            "C_dir": "CONFIRMED" if c_dir else "REFUTED",
            "C_notcm": "CONFIRMED" if c_notcm else "REFUTED",
            "C_null": "CONFIRMED" if c_nullg else "REFUTED",
        },
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_59_audit_direction.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
