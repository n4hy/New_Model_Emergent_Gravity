#!/usr/bin/env python3
"""M9.58 audit. N=10. Tries to REFUTE C_flat and C_class.

STAR: src (4,5,5), σ=0.9, α=0.03, R=2,3,4.
SEA: PBC band-edge, same R.
PAIR: A=(2,5,5), d=2,3,4.
Same a=−M/A and E_int.

Writes ../data/m9_58_audit_flux.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N = 10
SRC = (4, 5, 5)
SIG = 0.9
ALPHA = 0.03
RADII = (2, 3, 4)
SEPS = (2, 3, 4)
A0 = (2, 5, 5)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def outgoing_area(inside, n, pbc):
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
    return e1, v2 / np.linalg.norm(v2)


def energy_of(ham, corr):
    return float(np.sum(ham * corr))


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
    e_vac = energy_of(ham, c0)
    left, right = raw_packet(uo, uu, N, SRC, SIG)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    c1 = 0.5 * (
        (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left)))
        + (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))).T
    )
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    m_tot = float(np.sum(de))
    star = []
    for radius in RADII:
        inside = np.zeros((N, N, N), dtype=bool)
        sl = []
        cx, cy, cz = SRC
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                        inside[x, y, z] = True
                        sl.append(idx(x, y, z))
        area = outgoing_area(inside, N, False)
        mass = float(np.sum(de[np.array(sl, dtype=int)]))
        star.append(
            {
                "R": radius,
                "M": mass,
                "A": area,
                "a_A": -mass / area,
                "a_R2": -mass / float(radius * radius),
                "encl": mass / m_tot if m_tot else None,
            }
        )
    fit = [r for r in star if r["encl"] and r["encl"] > 0.95]
    sl_a = (
        slope_of([r["R"] for r in fit], [r["a_A"] for r in fit])
        if len(fit) >= 2
        else None
    )
    sl_r = (
        slope_of([r["R"] for r in fit], [r["a_R2"] for r in fit])
        if len(fit) >= 2
        else None
    )

    ham_u = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % N, (y + d[1]) % N, (z + d[2]) % N)
                    ham_u[i, j] = ham_u[j, i] = -1.0
    evu, vecu = np.linalg.eigh(ham_u)
    il, ir = int(np.argmin(evu)), int(np.argmax(evu))
    occu = evu < 0.0
    c0u = vecu[:, occu] @ vecu[:, occu].T
    dC = np.outer(vecu[:, ir], vecu[:, ir]) - np.outer(vecu[:, il], vecu[:, il])
    c1u = 0.5 * ((c0u + ALPHA * dC) + (c0u + ALPHA * dC).T)
    deu = np.sum(ham_u * c1u, axis=1) - np.sum(ham_u * c0u, axis=1)
    sea = []
    for radius in RADII:
        inside = np.zeros((N, N, N), dtype=bool)
        sl = []
        cx, cy, cz = SRC
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    dx = min((x - cx) % N, (cx - x) % N)
                    dy = min((y - cy) % N, (cy - y) % N)
                    dz = min((z - cz) % N, (cz - z) % N)
                    if dx * dx + dy * dy + dz * dz <= radius * radius:
                        inside[x, y, z] = True
                        sl.append(idx(x, y, z))
        area = outgoing_area(inside, N, True)
        mass = float(np.sum(deu[np.array(sl, dtype=int)]))
        sea.append(
            {
                "R": radius,
                "M": mass,
                "A": area,
                "a_A": -mass / area,
                "a_R2": -mass / float(radius * radius),
            }
        )
    sl_a_sea = slope_of(RADII, [r["a_A"] for r in sea])
    sl_r_sea = slope_of(RADII, [r["a_R2"] for r in sea])

    la0, ra0 = raw_packet(uo, uu, N, A0, SIG)
    la0 /= np.linalg.norm(la0)
    ra0 /= np.linalg.norm(ra0)
    e_one = energy_of(ham, c0 + ALPHA * (np.outer(ra0, ra0) - np.outer(la0, la0))) - e_vac
    pair = []
    rels = []
    for sep in SEPS:
        src_b = (A0[0] + sep, A0[1], A0[2])
        la, ra = raw_packet(uo, uu, N, A0, SIG)
        lb, rb = raw_packet(uo, uu, N, src_b, SIG)
        la, lb = orthonormalize(la, lb)
        ra, rb = orthonormalize(ra, rb)
        c_ab = 0.5 * (
            (
                c0
                + ALPHA * (np.outer(ra, ra) - np.outer(la, la))
                + ALPHA * (np.outer(rb, rb) - np.outer(lb, lb))
            )
            + (
                c0
                + ALPHA * (np.outer(ra, ra) - np.outer(la, la))
                + ALPHA * (np.outer(rb, rb) - np.outer(lb, lb))
            ).T
        )
        e_ab = energy_of(ham, c_ab) - e_vac
        lb1, rb1 = raw_packet(uo, uu, N, src_b, SIG)
        lb1 /= np.linalg.norm(lb1)
        rb1 /= np.linalg.norm(rb1)
        e_b = energy_of(ham, c0 + ALPHA * (np.outer(rb1, rb1) - np.outer(lb1, lb1))) - e_vac
        e_int = e_ab - e_one - e_b
        rel = abs(e_int) / (abs(e_one) + abs(e_b))
        rels.append(rel)
        pair.append({"d": sep, "E_int": e_int, "rel": rel})

    c_flat = bool(max(rels) < 0.02)
    c_star = sl_a is not None and abs(sl_a + 2.0) < 0.20
    c_sea = abs(sl_a_sea - 1.0) < abs(sl_a_sea + 2.0)
    payload = {
        "task": "m9.58_audit_flux",
        "star": star,
        "star_slope_A": sl_a,
        "star_slope_R2": sl_r,
        "sea": sea,
        "sea_slope_A": sl_a_sea,
        "sea_slope_R2": sl_r_sea,
        "pair": pair,
        "max_rel_Eint": float(max(rels)),
        "C_flat_PRIMARY": c_flat,
        "C_star": c_star,
        "C_sea": c_sea,
        "verdicts": {
            "C_flat": "CONFIRMED" if c_flat else "REFUTED",
            "C_star": "CONFIRMED" if c_star else "REFUTED",
            "C_sea": "CONFIRMED" if c_sea else "REFUTED",
        },
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_58_audit_flux.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
