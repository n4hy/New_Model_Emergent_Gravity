#!/usr/bin/env python3
"""M9.58: unique isotropic flux is g=−M/A. Does H contain 1/r?

Derivation (not a hop theorem). Entanglement supplies
M(B)=∑_{i∈B} δe. If a local g exists, Stokes says
∮_{∂B} g·dA = −M(B). On a source-centered ball,
isotropy forces g = g_R n-hat, so g_R = −M/A.
A is the outgoing nearest-neighbour count.

Paper 61 used −M/R². That equals the flux law iff
A ∝ R². This run uses A.

H is a nearest-neighbour hop. Two-packet
E_int(d)=E_AB−E_A−E_B is computed, not assumed.

PRE-REGISTERED:
  STAR: open N=12, src (6,6,6), σ=1, α=0.02.
        R=2,3,4,5. Slope of |a| vs R on R with
        M(R)/M_tot > 0.95 (need ≥3).
  SEA:  PBC band-edge, same R, all four.
  a_A = −M/A,  a_R2 = −M/R².
  C_star  |slope_A + 2| < 0.20
  C_sea   |slope_A − 1| < |slope_A + 2|
  C_class both prescriptions in the same class
          (star closer to −2 than +1; sea opposite)
  PAIR: open, A=(3,6,6), B=(3+d,6,6), d=2,3,4,5,6.
        Orthonormal two-source as M9.40.
  C_flat PRIMARY. max_d |E_int| / (|E_A|+|E_B|) < 0.02
          (no force in H at this threshold)

Not claimed: derived Einstein, 8πG, de Sitter, MODELS.md.
If C_flat passes, Gauss is the flux completion of M,
not a theorem of the hop.

Writes ../data/m9_58_flux.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N = 12
SRC = (6, 6, 6)
SIGMA = 1.0
ALPHA = 0.02
RADII = (2, 3, 4, 5)
SEPS = (2, 3, 4, 5, 6)
A0 = (3, 6, 6)


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


def hop_pbc(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % n, (y + d[1]) % n, (z + d[2]) % n, n)
                    ham[i, j] = ham[j, i] = -1.0
    return ham


def outgoing_area(inside, n, pbc: bool) -> int:
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
                    if pbc:
                        xx %= n
                        yy %= n
                        zz %= n
                    elif not (0 <= xx < n and 0 <= yy < n and 0 <= zz < n):
                        area += 1
                        continue
                    if not inside[xx, yy, zz]:
                        area += 1
    return int(area)


def ball_mask_open(center, radius, n):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                    inside[x, y, z] = True
                    sl.append(idx(x, y, z, n))
    return inside, np.array(sl, dtype=int)


def ball_mask_pbc(center, radius, n):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                dx = min((x - cx) % n, (cx - x) % n)
                dy = min((y - cy) % n, (cy - y) % n)
                dz = min((z - cz) % n, (cz - z) % n)
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    inside[x, y, z] = True
                    sl.append(idx(x, y, z, n))
    return inside, np.array(sl, dtype=int)


def slope_of(radii, accs):
    lr = np.log(np.asarray(radii, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([lr, np.ones(len(radii))]), la, rcond=None
    )
    return float(coef[0])


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


def energy_of(ham, corr):
    return float(np.sum(ham * corr))


def scan_force(de, masks):
    rows = []
    for radius, (inside, sl, area, pbc) in masks.items():
        mass = float(np.sum(de[sl]))
        a_a = -mass / area if area else float("nan")
        a_r = -mass / float(radius * radius)
        rows.append(
            {
                "R": radius,
                "M": mass,
                "A": area,
                "A_over_R2": area / float(radius * radius),
                "a_A": a_a,
                "a_R2": a_r,
            }
        )
    return rows


def class_star(slope):
    return abs(slope + 2.0) < abs(slope - 1.0)


def class_sea(slope):
    return abs(slope - 1.0) < abs(slope + 2.0)


def main() -> int:
    ham_s = hop_open(N)
    ev_s, vecs_s = np.linalg.eigh(ham_s)
    occ_s = ev_s < 0.0
    uo, uu = vecs_s[:, occ_s], vecs_s[:, ~occ_s]
    c0s = uo @ uo.T
    e0s = energy_of(ham_s, c0s)
    left, right = raw_packet(uo, uu, N, SRC, SIGMA)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c1s = 0.5 * (
        (c0s + ALPHA * (np.outer(right, right) - np.outer(left, left)))
        + (c0s + ALPHA * (np.outer(right, right) - np.outer(left, left))).T
    )
    de_s = np.sum(ham_s * c1s, axis=1) - np.sum(ham_s * c0s, axis=1)
    m_star = float(np.sum(de_s))

    star_masks = {}
    for radius in RADII:
        inside, sl = ball_mask_open(SRC, radius, N)
        area = outgoing_area(inside, N, False)
        star_masks[radius] = (inside, sl, area, False)
    star_rows = scan_force(de_s, star_masks)
    fit_r = [r["R"] for r in star_rows if m_star and r["M"] / m_star > 0.95]
    sl_a_star = (
        slope_of(fit_r, [r["a_A"] for r in star_rows if r["R"] in fit_r])
        if len(fit_r) >= 3
        else None
    )
    sl_r_star = (
        slope_of(fit_r, [r["a_R2"] for r in star_rows if r["R"] in fit_r])
        if len(fit_r) >= 3
        else None
    )

    ham_u = hop_pbc(N)
    ev_u, vecs_u = np.linalg.eigh(ham_u)
    il, ir = int(np.argmin(ev_u)), int(np.argmax(ev_u))
    occ_u = ev_u < 0.0
    c0u = vecs_u[:, occ_u] @ vecs_u[:, occ_u].T
    dC_u = np.outer(vecs_u[:, ir], vecs_u[:, ir]) - np.outer(
        vecs_u[:, il], vecs_u[:, il]
    )
    c1u = 0.5 * ((c0u + ALPHA * dC_u) + (c0u + ALPHA * dC_u).T)
    de_u = np.sum(ham_u * c1u, axis=1) - np.sum(ham_u * c0u, axis=1)
    sea_masks = {}
    for radius in RADII:
        inside, sl = ball_mask_pbc(SRC, radius, N)
        area = outgoing_area(inside, N, True)
        sea_masks[radius] = (inside, sl, area, True)
    sea_rows = scan_force(de_u, sea_masks)
    sl_a_sea = slope_of(RADII, [r["a_A"] for r in sea_rows])
    sl_r_sea = slope_of(RADII, [r["a_R2"] for r in sea_rows])

    c_star = sl_a_star is not None and abs(sl_a_star + 2.0) < 0.20
    c_sea = abs(sl_a_sea - 1.0) < abs(sl_a_sea + 2.0)
    c_class = bool(
        sl_a_star is not None
        and sl_r_star is not None
        and class_star(sl_a_star)
        and class_star(sl_r_star)
        and class_sea(sl_a_sea)
        and class_sea(sl_r_sea)
    )

    la0, ra0 = raw_packet(uo, uu, N, A0, SIGMA)
    la0 = la0 / np.linalg.norm(la0)
    ra0 = ra0 / np.linalg.norm(ra0)
    e_vac = e0s
    e_one = (
        energy_of(ham_s, c0s + ALPHA * (np.outer(ra0, ra0) - np.outer(la0, la0)))
        - e_vac
    )
    pair_rows = []
    rels = []
    for sep in SEPS:
        src_b = (A0[0] + sep, A0[1], A0[2])
        la, ra = raw_packet(uo, uu, N, A0, SIGMA)
        lb, rb = raw_packet(uo, uu, N, src_b, SIGMA)
        la, lb = orthonormalize(la, lb)
        ra, rb = orthonormalize(ra, rb)
        c_ab = 0.5 * (
            (
                c0s
                + ALPHA * (np.outer(ra, ra) - np.outer(la, la))
                + ALPHA * (np.outer(rb, rb) - np.outer(lb, lb))
            )
            + (
                c0s
                + ALPHA * (np.outer(ra, ra) - np.outer(la, la))
                + ALPHA * (np.outer(rb, rb) - np.outer(lb, lb))
            ).T
        )
        e_ab = energy_of(ham_s, c_ab) - e_vac
        lb1, rb1 = raw_packet(uo, uu, N, src_b, SIGMA)
        lb1 = lb1 / np.linalg.norm(lb1)
        rb1 = rb1 / np.linalg.norm(rb1)
        e_b = (
            energy_of(ham_s, c0s + ALPHA * (np.outer(rb1, rb1) - np.outer(lb1, lb1)))
            - e_vac
        )
        e_int = e_ab - e_one - e_b
        rel = abs(e_int) / (abs(e_one) + abs(e_b)) if (e_one or e_b) else float("nan")
        rels.append(rel)
        pair_rows.append(
            {
                "d": sep,
                "E_AB": e_ab,
                "E_A": e_one,
                "E_B": e_b,
                "E_int": e_int,
                "rel": rel,
            }
        )
    c_flat = bool(max(rels) < 0.02)

    ok = bool(c_star and c_sea and c_class and c_flat)
    payload = {
        "task": "m9.58_flux",
        "derivation": "Stokes + isotropy => g_R = -M/A; M from entanglement",
        "star": {
            "rows": star_rows,
            "fit_R": fit_r,
            "slope_A": sl_a_star,
            "slope_R2": sl_r_star,
        },
        "sea": {
            "rows": sea_rows,
            "slope_A": sl_a_sea,
            "slope_R2": sl_r_sea,
        },
        "pair": pair_rows,
        "C_star": c_star,
        "C_sea": c_sea,
        "C_class": c_class,
        "C_flat_PRIMARY": c_flat,
        "max_rel_Eint": float(max(rels)),
        "all_gates": ok,
        "verdict": "FLUX_GAUSS_H_FLAT" if ok else "FLUX_OR_H_FAIL",
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_58_flux.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
