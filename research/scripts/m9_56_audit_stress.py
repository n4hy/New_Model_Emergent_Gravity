#!/usr/bin/env python3
"""M9.56 audit. Own lattices; tries to REFUTE C_dust and to
CONFIRM C_lambda FAIL.

N=10. STAR: open, src (4,5,5), σ=0.9.
SEA: PBC band edges. α ∈ {0.015, 0.05}.
Same virial: w=(2/3) k·sin k, k folded to (−π,π].

Writes ../data/m9_56_audit_stress.json
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
ALPHAS = (0.015, 0.05)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


def fold(k):
    return float((k + np.pi) % (2.0 * np.pi) - np.pi)


def virial(kx, ky, kz):
    return (2.0 / 3.0) * (
        kx * np.sin(kx) + ky * np.sin(ky) + kz * np.sin(kz)
    )


def hop_eps(kx, ky, kz):
    return float(-2.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz)))


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
    env = np.zeros(N**3)
    stag = np.zeros(N**3)
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (SIG * SIG))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    nn = np.arange(1, N + 1, dtype=float)
    k1 = np.pi * nn / (N + 1.0)
    e1 = -2.0 * np.cos(k1)
    j = np.arange(1, N + 1, dtype=float)
    psi = np.sqrt(2.0 / (N + 1.0)) * np.sin(np.outer(j, k1))
    l3 = left.reshape(N, N, N)
    r3 = right.reshape(N, N, N)
    amp_l = np.einsum("xa,yb,zc,xyz->abc", psi, psi, psi, l3)
    amp_r = np.einsum("xa,yb,zc,xyz->abc", psi, psi, psi, r3)
    kx = k1[:, None, None]
    ky = k1[None, :, None]
    kz = k1[None, None, :]
    eps = e1[:, None, None] + e1[None, :, None] + e1[None, None, :]
    w = virial(kx, ky, kz)
    dn1 = amp_r**2 - amp_l**2
    d_e1 = float(np.sum(dn1 * eps))
    d_p1 = float(np.sum(dn1 * w))
    site1 = float(right @ (ham @ right) - left @ (ham @ left))
    w_l = virial(0.0, 0.0, 0.0)
    w_r = virial(np.pi, np.pi, np.pi)
    e_l, e_r = hop_eps(0.0, 0.0, 0.0), hop_eps(np.pi, np.pi, np.pi)
    rows = []
    c_lambda = True
    c_dust = True
    c_e = abs(d_e1 - site1) / abs(site1) < 1e-8
    for alpha in ALPHAS:
        r_star = (alpha * d_p1) / (alpha * d_e1)
        r_sea = (alpha * (w_r - w_l)) / (alpha * (e_r - e_l))
        if abs(r_sea + 1.0) >= 0.25:
            c_lambda = False
        if abs(r_sea) >= 0.25:
            c_dust = False
        rows.append(
            {
                "alpha": alpha,
                "star_r": r_star,
                "sea_r": r_sea,
                "sea_w_L": w_l,
                "sea_w_R": w_r,
            }
        )
    payload = {
        "task": "m9.56_audit_stress",
        "C_e": c_e,
        "star_site_rel_err": abs(d_e1 - site1) / abs(site1),
        "rows": rows,
        "C_lambda_PRIMARY": c_lambda,
        "C_dust": c_dust,
        "verdicts": {
            "C_lambda": "CONFIRMED" if c_lambda else "REFUTED",
            "C_dust": "CONFIRMED" if c_dust else "REFUTED",
            "C_e": "CONFIRMED" if c_e else "REFUTED",
        },
        "not_claimed": ["8pi G", "derived Einstein", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_56_audit_stress.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
