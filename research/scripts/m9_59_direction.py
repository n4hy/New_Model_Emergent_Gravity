#!/usr/bin/env python3
"""M9.59: two-mass direction on the entangled pair.

    n(c) = ∇_c M_AB(c) / |∇_c M_AB(c)|
    g(c) = − (M_AB(c) / A(c)) n(c)

∇_c is the exact two-sided lattice difference of the
mass map, not a continuum approximation.

eigh is float64 LAPACK (only practical  N³ solver).
Every mass, gradient, Newton vector, angle, and gate
is mpmath dps=50. Floors are fractions of M_AB, not
absolute epsilons.

PRE-REGISTERED:
  N=12 open. A=(3,6,6), B=(9,6,6), σ=1.
  Unequal: α_A=0.02, α_B=0.04. Orthonormal two-source.
  R=3. Two-sided centres only.
  Keep if |∇M| > 1e-6 M_AB and 0.15 < M/M_AB < 0.92.
  C_dir PRIMARY. median angle(n, g_N) < 20°
  C_notcm  median angle(n, g_N) < median angle(n, u_cm)
  C_null   |∇M| nearer the 1/r² null < |∇M| at CM
  Equal α=0.02 both: C_mid |∇M|(mid) / median|∇M| < 0.15
  C_add    |M_AB − M_A − M_B| / M_AB < 1e-4
           (pair energy additivity; numerical hygiene)

Not claimed: derived Einstein, 8πG, de Sitter, MODELS.md.

Writes ../data/m9_59_direction.json
"""

from __future__ import annotations

import json
import os

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
mp.mp.dps = 50
N = 12
SRC_A = (3, 6, 6)
SRC_B = (9, 6, 6)
MID = (6, 6, 6)
SIGMA = 1.0
ALPHA_A = 0.02
ALPHA_B = 0.04
RADIUS = 3
FRAC_LO = mp.mpf("0.15")
FRAC_HI = mp.mpf("0.92")
GRAD_FRAC = mp.mpf("1e-6")


def mpf(x):
    return mp.mpf(x) if not isinstance(x, mp.mpf) else x


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_open(n: int) -> np.ndarray:
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
    n2 = np.linalg.norm(v2)
    if n2 < 1e-14:
        raise RuntimeError("packets linearly dependent")
    return e1, v2 / n2


def de_to_mp(de):
    return [mpf(x) for x in de]


def outgoing_area(inside, n) -> int:
    area = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if not inside[x, y, z]:
                    continue
                for d in (
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if not (0 <= xx < n and 0 <= yy < n and 0 <= zz < n):
                        area += 1
                        continue
                    if not inside[xx, yy, zz]:
                        area += 1
    return int(area)


def ball_sl(center, radius, n):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                    inside[x, y, z] = True
                    sl.append(idx(x, y, z, n))
    return inside, sl


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


def pair_de(uo, uu, ham, c0, e0, src_a, src_b, sigma, alpha_a, alpha_b):
    la, ra = raw_packet(uo, uu, N, src_a, sigma)
    lb, rb = raw_packet(uo, uu, N, src_b, sigma)
    la, lb = orthonormalize(la, lb)
    ra, rb = orthonormalize(ra, rb)
    corr = (
        c0
        + alpha_a * (np.outer(ra, ra) - np.outer(la, la))
        + alpha_b * (np.outer(rb, rb) - np.outer(lb, lb))
    )
    corr = 0.5 * (corr + corr.T)
    return np.sum(ham * corr, axis=1) - e0


def one_de(uo, uu, ham, c0, e0, src, sigma, alpha):
    left, right = raw_packet(uo, uu, N, src, sigma)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    corr = 0.5 * (corr + corr.T)
    return np.sum(ham * corr, axis=1) - e0


def mass_map(de_mp, radius, n):
    lo, hi = radius, n - radius
    centres = [
        (x, y, z)
        for x in range(lo, hi)
        for y in range(lo, hi)
        for z in range(lo, hi)
    ]
    areas = {}
    masses = {}
    for c in centres:
        inside, sl = ball_sl(c, radius, n)
        areas[c] = areas.get(c, outgoing_area(inside, n))
        acc = mp.mpf(0)
        for i in sl:
            acc += de_mp[i]
        masses[c] = acc
    return centres, areas, masses


def grad_at(c, masses):
    g = []
    for ax in range(3):
        cp = list(c)
        cm = list(c)
        cp[ax] += 1
        cm[ax] -= 1
        tp, tm = tuple(cp), tuple(cm)
        if tp not in masses or tm not in masses:
            return None
        g.append((masses[tp] - masses[tm]) / 2)
    return g


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


def score_pair(masses, areas, m_a, m_b, m_tot, pos_a, pos_b):
    r_cm = [(m_a * pos_a[i] + m_b * pos_b[i]) / (m_a + m_b) for i in range(3)]
    null_x = invsq_null_x(pos_a[0], pos_b[0], m_a, m_b)
    floor = GRAD_FRAC * abs(m_tot)
    angs_n = []
    angs_cm = []
    recs = []
    for c, mass in masses.items():
        gm = grad_at(c, masses)
        if gm is None:
            continue
        gn = mp.sqrt(sum(x * x for x in gm))
        frac = mass / m_tot if m_tot != 0 else mp.mpf(0)
        if gn <= floor or frac <= FRAC_LO or frac >= FRAC_HI:
            continue
        g_n = newton_g(c, pos_a, pos_b, m_a, m_b)
        u_cm = [r_cm[i] - mpf(c[i]) for i in range(3)]
        ang_n = angle_deg(gm, g_n)
        ang_cm = angle_deg(gm, u_cm)
        if ang_n is None:
            continue
        angs_n.append(ang_n)
        if ang_cm is not None:
            angs_cm.append(ang_cm)
        recs.append(
            {
                "c": list(c),
                "M": fnum(mass),
                "frac": fnum(frac),
                "grad_norm": fnum(gn),
                "ang_newton": fnum(ang_n),
                "ang_cm": fnum(ang_cm),
                "A": areas[c],
            }
        )
    med_n = mp_median(angs_n)
    med_cm = mp_median(angs_cm)
    axis = [
        c
        for c in masses
        if c[1] == 6 and c[2] == 6 and grad_at(c, masses) is not None
    ]
    c_null = min(axis, key=lambda c: abs(mpf(c[0]) - null_x)) if axis else None
    c_cm = min(axis, key=lambda c: abs(mpf(c[0]) - r_cm[0])) if axis else None
    g_null = (
        mp.sqrt(sum(x * x for x in grad_at(c_null, masses))) if c_null else None
    )
    g_cmv = mp.sqrt(sum(x * x for x in grad_at(c_cm, masses))) if c_cm else None
    return {
        "n_keep": len(recs),
        "median_ang_newton": fnum(med_n),
        "median_ang_cm": fnum(med_cm),
        "null_x": fnum(null_x),
        "cm_x": fnum(r_cm[0]),
        "c_null": list(c_null) if c_null else None,
        "c_cm": list(c_cm) if c_cm else None,
        "grad_at_null": fnum(g_null),
        "grad_at_cm": fnum(g_cmv),
        "sample": recs[:8],
        "_med_n": med_n,
        "_med_cm": med_cm,
        "_g_null": g_null,
        "_g_cm": g_cmv,
    }


def main() -> int:
    ham = hop_open(N)
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T
    e0 = np.sum(ham * c0, axis=1)
    de_a = de_to_mp(one_de(uo, uu, ham, c0, e0, SRC_A, SIGMA, ALPHA_A))
    de_b = de_to_mp(one_de(uo, uu, ham, c0, e0, SRC_B, SIGMA, ALPHA_B))
    de_ab = de_to_mp(
        pair_de(uo, uu, ham, c0, e0, SRC_A, SRC_B, SIGMA, ALPHA_A, ALPHA_B)
    )
    m_a = sum(de_a, mp.mpf(0))
    m_b = sum(de_b, mp.mpf(0))
    m_tot = sum(de_ab, mp.mpf(0))
    add_rel = abs(m_tot - m_a - m_b) / abs(m_tot)
    pos_a = [mpf(x) for x in SRC_A]
    pos_b = [mpf(x) for x in SRC_B]
    _c, areas, masses = mass_map(de_ab, RADIUS, N)
    uneq = score_pair(masses, areas, m_a, m_b, m_tot, pos_a, pos_b)

    de_eq = de_to_mp(
        pair_de(uo, uu, ham, c0, e0, SRC_A, SRC_B, SIGMA, ALPHA_A, ALPHA_A)
    )
    _c2, _a2, mass_eq = mass_map(de_eq, RADIUS, N)
    g_mid = grad_at(MID, mass_eq)
    g_mags = []
    for c in mass_eq:
        gm = grad_at(c, mass_eq)
        if gm is not None:
            g_mags.append(mp.sqrt(sum(x * x for x in gm)))
    med_g = mp_median(g_mags)
    g_mid_n = mp.sqrt(sum(x * x for x in g_mid)) if g_mid is not None else None
    mid_ratio = g_mid_n / med_g if (g_mid_n is not None and med_g) else None

    c_dir = bool(
        uneq["n_keep"] >= 8
        and uneq["_med_n"] is not None
        and uneq["_med_n"] < 20
    )
    c_notcm = bool(
        uneq["n_keep"] >= 8
        and uneq["_med_n"] is not None
        and uneq["_med_cm"] is not None
        and uneq["_med_n"] < uneq["_med_cm"]
    )
    c_null = bool(
        uneq["_g_null"] is not None
        and uneq["_g_cm"] is not None
        and uneq["_g_null"] < uneq["_g_cm"]
    )
    c_mid = bool(mid_ratio is not None and mid_ratio < mp.mpf("0.15"))
    c_add = bool(add_rel < mp.mpf("1e-4"))
    for key in ("_med_n", "_med_cm", "_g_null", "_g_cm"):
        uneq.pop(key, None)
    ok = bool(c_dir and c_notcm and c_null and c_mid and c_add)
    payload = {
        "task": "m9.59_direction",
        "equation": "n = lattice_grad_c M_AB; g = -(M/A) n",
        "precision": {
            "eigh": "float64 LAPACK",
            "analysis": "mpmath dps=50",
            "grad": "exact two-sided lattice difference",
            "grad_floor": "1e-6 M_AB",
            "additivity_rel": fnum(add_rel),
        },
        "M_A": fnum(m_a),
        "M_B": fnum(m_b),
        "M_AB": fnum(m_tot),
        "unequal": uneq,
        "equal_mid": {
            "grad_at_mid": fnum(g_mid_n),
            "median_grad": fnum(med_g),
            "ratio": fnum(mid_ratio),
        },
        "C_dir_PRIMARY": c_dir,
        "C_notcm": c_notcm,
        "C_null": c_null,
        "C_mid": c_mid,
        "C_add": c_add,
        "all_gates": ok,
        "verdict": "PAIR_DIRECTION" if ok else "PAIR_DIRECTION_FAIL",
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_59_direction.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
