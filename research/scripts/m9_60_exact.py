#!/usr/bin/env python3
"""M9.60: pair direction from the exact open-hop basis.

Paper 69 used float64 LAPACK for H. The open cube hop
is separable. 1D modes are closed form:

    k_n = π n/(N+1),   ε_n = −2 cos k_n,
    ψ_j^{(n)} = √(2/(N+1)) sin(k_n (j+1)),  n,j = 1..N.

No eigensolve. Occupied ⇔ ε_x+ε_y+ε_z < 0.
Every amplitude, mass, gradient, Newton vector,
and angle is mpmath dps=80.

Same pair equation as Paper 69:

    n(c) = ∇_c M_AB / |∇_c M_AB|
    g(c) = −(M_AB/A) n(c)

PRE-REGISTERED:
  N=12. A=(3,6,6), B=(9,6,6), σ=1.
  α_A=0.02, α_B=0.04. R=3.
  Floor |∇M| > 1e-6 M_AB, 0.15 < f < 0.92.
  C_dir PRIMARY. median angle(n, g_N) < 20°
  C_notcm  median angle(n, g_N) < median angle(n, u_cm)
  C_null   |∇M| at 1/r²-null site < |∇M| at CM site
  C_mid    equal pair, |∇M|(mid)/median < 0.15
  C_add    |M_AB−M_A−M_B|/M_AB < 1e-12
           (stricter: no LAPACK residual)
  C_cert   |θ_dir − 10.75°| < 1.0
           (Paper 69 angle was physics, not LAPACK)

Not claimed: derived Einstein, 8πG, de Sitter, MODELS.md.

Writes ../data/m9_60_exact.json
"""

from __future__ import annotations

import json
import os

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
mp.mp.dps = 80
N = 12
SRC_A = (3, 6, 6)
SRC_B = (9, 6, 6)
MID = (6, 6, 6)
SIGMA = mp.mpf(1)
ALPHA_A = mp.mpf("0.02")
ALPHA_B = mp.mpf("0.04")
RADIUS = 3
FRAC_LO = mp.mpf("0.15")
FRAC_HI = mp.mpf("0.92")
GRAD_FRAC = mp.mpf("1e-6")
PAPER69_DIR = mp.mpf("10.75")


def idx(x, y, z, n=None):
    if n is None:
        n = N
    return (x * n + y) * n + z


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


def open_1d(n):
    psi = [[mp.mpf(0) for _ in range(n)] for _ in range(n)]
    eps = []
    norm = mp.sqrt(2 / mp.mpf(n + 1))
    for a in range(n):
        kn = mp.pi * (a + 1) / (n + 1)
        eps.append(-2 * mp.cos(kn))
        for j in range(n):
            psi[j][a] = norm * mp.sin(kn * (j + 1))
    return psi, eps


def zeros3(n):
    return [
        [[mp.mpf(0) for _ in range(n)] for _ in range(n)] for _ in range(n)
    ]


def add_scaled(out, src, scale):
    n = len(out)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                out[x][y][z] += scale * src[x][y][z]


def contract_amp(psi, eps, field):
    n = len(eps)
    tmp_x = zeros3(n)
    for nx in range(n):
        for y in range(n):
            for z in range(n):
                acc = mp.mpf(0)
                for x in range(n):
                    acc += psi[x][nx] * field[x][y][z]
                tmp_x[nx][y][z] = acc
    tmp_xy = zeros3(n)
    for nx in range(n):
        for ny in range(n):
            for z in range(n):
                acc = mp.mpf(0)
                for y in range(n):
                    acc += psi[y][ny] * tmp_x[nx][y][z]
                tmp_xy[nx][ny][z] = acc
    amp = [
        [[mp.mpf(0) for _ in range(n)] for _ in range(n)] for _ in range(n)
    ]
    for nx in range(n):
        for ny in range(n):
            for nz in range(n):
                acc = mp.mpf(0)
                for z in range(n):
                    acc += psi[z][nz] * tmp_xy[nx][ny][z]
                amp[nx][ny][nz] = acc
    return amp


def synthesize(psi, eps, amp, occupied):
    n = len(eps)
    # only occupied (or only unoccupied) amplitudes kept
    tmp_xy = zeros3(n)
    for nx in range(n):
        for ny in range(n):
            for z in range(n):
                acc = mp.mpf(0)
                for nz in range(n):
                    if occupied[nx][ny][nz]:
                        acc += amp[nx][ny][nz] * psi[z][nz]
                tmp_xy[nx][ny][z] = acc
    tmp_x = zeros3(n)
    for nx in range(n):
        for y in range(n):
            for z in range(n):
                acc = mp.mpf(0)
                for ny in range(n):
                    acc += tmp_xy[nx][ny][z] * psi[y][ny]
                tmp_x[nx][y][z] = acc
    out = zeros3(n)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                acc = mp.mpf(0)
                for nx in range(n):
                    acc += tmp_x[nx][y][z] * psi[x][nx]
                out[x][y][z] = acc
    return out


def occupy_mask(eps):
    n = len(eps)
    mask = [
        [[False for _ in range(n)] for _ in range(n)] for _ in range(n)
    ]
    n_occ = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if eps[a] + eps[b] + eps[c] < 0:
                    mask[a][b][c] = True
                    n_occ += 1
    return mask, n_occ


def gaussian(n, src, sigma):
    env = zeros3(n)
    stag = zeros3(n)
    sx, sy, sz = src
    half = mp.mpf("0.5")
    for x in range(n):
        for y in range(n):
            for z in range(n):
                rr = mp.mpf((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)
                val = mp.e ** (-half * rr / (sigma * sigma))
                env[x][y][z] = val
                stag[x][y][z] = (mp.mpf(1) - 2 * ((x + y + z) % 2)) * val
    return env, stag


def nrm(field):
    acc = mp.mpf(0)
    n = len(field)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                acc += field[x][y][z] ** 2
    return mp.sqrt(acc)


def scale_inplace(field, s):
    n = len(field)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                field[x][y][z] *= s


def dot3(a, b):
    acc = mp.mpf(0)
    n = len(a)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                acc += a[x][y][z] * b[x][y][z]
    return acc


def axpy(out, src, s):
    n = len(out)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                out[x][y][z] += s * src[x][y][z]


def copy3(field):
    n = len(field)
    return [[list(field[x][y]) for y in range(n)] for x in range(n)]


def project_occ(psi, eps, mask, field):
    amp = contract_amp(psi, eps, field)
    return synthesize(psi, eps, amp, mask)


def packet_lr(psi, eps, mask, src, sigma):
    env, stag = gaussian(N, src, sigma)
    left = project_occ(psi, eps, mask, env)
    occ_stag = project_occ(psi, eps, mask, stag)
    right = copy3(stag)
    axpy(right, occ_stag, mp.mpf(-1))
    nl, nr = nrm(left), nrm(right)
    if nl == 0 or nr == 0:
        raise RuntimeError("vanishing packet")
    scale_inplace(left, 1 / nl)
    scale_inplace(right, 1 / nr)
    return left, right


def orthonormalize(a1, a2):
    e1 = copy3(a1)
    scale_inplace(e1, 1 / nrm(e1))
    e2 = copy3(a2)
    axpy(e2, e1, -dot3(e1, a2))
    n2 = nrm(e2)
    if n2 == 0:
        raise RuntimeError("packets linearly dependent")
    scale_inplace(e2, 1 / n2)
    return e1, e2


def site_de(lefts_rights_alphas):
    """Δe_i = −∑_{j nn i} ΔC_ij, ΔC from rank-2 updates."""
    n = N
    de = [mp.mpf(0) for _ in range(n**3)]
    neigh = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z)
                acc = mp.mpf(0)
                for d in neigh:
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if not (0 <= xx < n and 0 <= yy < n and 0 <= zz < n):
                        continue
                    dc = mp.mpf(0)
                    for left, right, alpha in lefts_rights_alphas:
                        dc += alpha * (
                            right[x][y][z] * right[xx][yy][zz]
                            - left[x][y][z] * left[xx][yy][zz]
                        )
                    acc -= dc
                de[i] = acc
    return de


def outgoing_area(inside, n):
    area = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if not inside[x][y][z]:
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
                    if not inside[xx][yy][zz]:
                        area += 1
    return int(area)


def ball_inside(center, radius, n):
    cx, cy, cz = center
    inside = [[[False] * n for _ in range(n)] for _ in range(n)]
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                    inside[x][y][z] = True
                    sl.append(idx(x, y, z, n))
    return inside, sl


def mass_map(de, radius, n):
    lo, hi = radius, n - radius
    masses, areas = {}, {}
    for x in range(lo, hi):
        for y in range(lo, hi):
            for z in range(lo, hi):
                c = (x, y, z)
                inside, sl = ball_inside(c, radius, n)
                areas[c] = outgoing_area(inside, n)
                acc = mp.mpf(0)
                for i in sl:
                    acc += de[i]
                masses[c] = acc
    return masses, areas


def grad_at(c, masses):
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
        d = [pos[i] - mp.mpf(c[i]) for i in range(3)]
        r2 = sum(x * x for x in d)
        if r2 == 0:
            continue
        r = mp.sqrt(r2)
        fac = mass / (r2 * r)
        for i in range(3):
            g[i] += fac * d[i]
    return g


def invsq_null_x(x_a, x_b, m_a, m_b):
    length = mp.mpf(x_b) - mp.mpf(x_a)
    return mp.mpf(x_a) + length / (1 + mp.sqrt(m_b / m_a))


def score_pair(masses, m_a, m_b, m_tot, pos_a, pos_b):
    r_cm = [(m_a * pos_a[i] + m_b * pos_b[i]) / (m_a + m_b) for i in range(3)]
    null_x = invsq_null_x(pos_a[0], pos_b[0], m_a, m_b)
    floor = GRAD_FRAC * abs(m_tot)
    angs_n, angs_cm, gn_list, gN_list = [], [], [], []
    recs = []
    for c, mass in masses.items():
        gm = grad_at(c, masses)
        if gm is None:
            continue
        gn = mp.sqrt(sum(x * x for x in gm))
        frac = mass / m_tot
        if gn <= floor or frac <= FRAC_LO or frac >= FRAC_HI:
            continue
        g_n = newton_g(c, pos_a, pos_b, m_a, m_b)
        u_cm = [r_cm[i] - mp.mpf(c[i]) for i in range(3)]
        ang_n = angle_deg(gm, g_n)
        ang_cm = angle_deg(gm, u_cm)
        if ang_n is None:
            continue
        angs_n.append(ang_n)
        if ang_cm is not None:
            angs_cm.append(ang_cm)
        gn_list.append(gn)
        gN_list.append(mp.sqrt(sum(x * x for x in g_n)))
        recs.append(
            {
                "c": list(c),
                "M": fnum(mass),
                "frac": fnum(frac),
                "grad_norm": fnum(gn),
                "ang_newton": fnum(ang_n),
                "ang_cm": fnum(ang_cm),
            }
        )
    med_n, med_cm = mp_median(angs_n), mp_median(angs_cm)
    axis = [
        c
        for c in masses
        if c[1] == 6 and c[2] == 6 and grad_at(c, masses) is not None
    ]
    c_null = min(axis, key=lambda c: abs(mp.mpf(c[0]) - null_x)) if axis else None
    c_cm = min(axis, key=lambda c: abs(mp.mpf(c[0]) - r_cm[0])) if axis else None
    g_null = (
        mp.sqrt(sum(x * x for x in grad_at(c_null, masses))) if c_null else None
    )
    g_cm = (
        mp.sqrt(sum(x * x for x in grad_at(c_cm, masses))) if c_cm else None
    )
    return {
        "n_keep": len(recs),
        "median_ang_newton": fnum(med_n),
        "median_ang_cm": fnum(med_cm),
        "null_x": fnum(null_x),
        "cm_x": fnum(r_cm[0]),
        "c_null": list(c_null) if c_null else None,
        "c_cm": list(c_cm) if c_cm else None,
        "grad_at_null": fnum(g_null),
        "grad_at_cm": fnum(g_cm),
        "sample": recs[:6],
        "_med_n": med_n,
        "_med_cm": med_cm,
        "_g_null": g_null,
        "_g_cm": g_cm,
    }


def main() -> int:
    psi, eps = open_1d(N)
    mask, n_occ = occupy_mask(eps)
    la, ra = packet_lr(psi, eps, mask, SRC_A, SIGMA)
    lb, rb = packet_lr(psi, eps, mask, SRC_B, SIGMA)
    # singles
    de_a = site_de([(la, ra, ALPHA_A)])
    de_b = site_de([(lb, rb, ALPHA_B)])
    # pair, orthonormalize across sources
    la_p, lb_p = orthonormalize(la, lb)
    ra_p, rb_p = orthonormalize(ra, rb)
    de_ab = site_de([(la_p, ra_p, ALPHA_A), (lb_p, rb_p, ALPHA_B)])
    m_a = sum(de_a, mp.mpf(0))
    m_b = sum(de_b, mp.mpf(0))
    m_tot = sum(de_ab, mp.mpf(0))
    add_rel = abs(m_tot - m_a - m_b) / abs(m_tot)
    pos_a = [mp.mpf(x) for x in SRC_A]
    pos_b = [mp.mpf(x) for x in SRC_B]
    masses, _areas = mass_map(de_ab, RADIUS, N)
    uneq = score_pair(masses, m_a, m_b, m_tot, pos_a, pos_b)

    de_eq = site_de([(la_p, ra_p, ALPHA_A), (lb_p, rb_p, ALPHA_A)])
    mass_eq, _ = mass_map(de_eq, RADIUS, N)
    g_mid = grad_at(MID, mass_eq)
    g_mags = []
    for c in mass_eq:
        gm = grad_at(c, mass_eq)
        if gm is not None:
            g_mags.append(mp.sqrt(sum(x * x for x in gm)))
    med_g = mp_median(g_mags)
    g_mid_n = mp.sqrt(sum(x * x for x in g_mid)) if g_mid is not None else None
    mid_ratio = g_mid_n / med_g if (g_mid_n is not None and med_g) else None

    theta = uneq["_med_n"]
    c_dir = bool(uneq["n_keep"] >= 8 and theta is not None and theta < 20)
    c_notcm = bool(
        theta is not None and uneq["_med_cm"] is not None and theta < uneq["_med_cm"]
    )
    c_null = bool(
        uneq["_g_null"] is not None
        and uneq["_g_cm"] is not None
        and uneq["_g_null"] < uneq["_g_cm"]
    )
    c_mid = bool(mid_ratio is not None and mid_ratio < mp.mpf("0.15"))
    c_add = bool(add_rel < mp.mpf("1e-12"))
    c_cert = bool(theta is not None and abs(theta - PAPER69_DIR) < 1)
    for key in ("_med_n", "_med_cm", "_g_null", "_g_cm"):
        uneq.pop(key, None)
    ok = bool(c_dir and c_notcm and c_null and c_mid and c_add)
    payload = {
        "task": "m9.60_exact",
        "equation": "n = lattice_grad_c M_AB; g = -(M/A) n; exact open-hop basis",
        "precision": {
            "basis": "closed-form 1d open hop, Kronecker 3d",
            "eigh": "none",
            "analysis": "mpmath dps=80",
            "n_occ": n_occ,
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
        "theta_minus_paper69": fnum(theta - PAPER69_DIR) if theta is not None else None,
        "C_dir_PRIMARY": c_dir,
        "C_notcm": c_notcm,
        "C_null": c_null,
        "C_mid": c_mid,
        "C_add": c_add,
        "C_cert": c_cert,
        "all_gates": ok,
        "verdict": "EXACT_PAIR_DIRECTION" if ok else "EXACT_PAIR_FAIL",
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_60_exact.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
