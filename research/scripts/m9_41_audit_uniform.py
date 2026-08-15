#!/usr/bin/env python3
"""M9.41 audit. N=10, source (4,5,5), σ_wide=6, σ_comp=0.9, α=0.03.

Own volume scan and own fill-cube Poisson. Tries to REFUTE
C_vol and C_lin.

Writes ../data/m9_41_audit_uniform.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, SRC, ALPHA = 10, (4, 5, 5), 0.03
SIG_W, SIG_C = 6.0, 0.9
RADII = (2, 3, 4)
LBOX, NBOX, GCONST = 1.0, 65, 1.0
PROBES = (0.10 * LBOX, 0.15 * LBOX, 0.20 * LBOX)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


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
                        ham[i, idx(xx, yy, zz)] = ham[idx(xx, yy, zz), i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    c0 = uo @ uo.T

    def transfer(sigma):
        env = np.zeros(vol)
        stag = np.zeros(vol)
        sx, sy, sz = SRC
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i = idx(x, y, z)
                    rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                    env[i] = np.exp(-0.5 * rr / (sigma * sigma))
                    stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
        left = uo @ (uo.T @ env)
        right = uu @ (uu.T @ stag)
        left = left / np.linalg.norm(left)
        right = right / np.linalg.norm(right)
        corr = c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))
        return 0.5 * (corr + corr.T)

    cw, cc = transfer(SIG_W), transfer(SIG_C)
    dew = np.sum(ham * cw, axis=1) - np.sum(ham * c0, axis=1)
    dec = np.sum(ham * cc, axis=1) - np.sum(ham * c0, axis=1)

    def scan(c1, de):
        ds, pf, vv, aa = [], [], [], []
        sx, sy, sz = SRC
        for rad in RADII:
            inside = np.zeros((N, N, N), dtype=bool)
            sl = []
            for x in range(N):
                for y in range(N):
                    for z in range(N):
                        if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= rad * rad:
                            inside[x, y, z] = True
                            sl.append(idx(x, y, z))
            sl = np.array(sl, dtype=int)
            area = 0
            for x in range(N):
                for y in range(N):
                    for z in range(N):
                        if not inside[x, y, z]:
                            continue
                        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                            xx, yy, zz = x + d[0], y + d[1], z + d[2]
                            if not (0 <= xx < N and 0 <= yy < N and 0 <= zz < N) or not inside[xx, yy, zz]:
                                area += 1
            ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
            pf.append(float(np.sum(de[sl])))
            vv.append(int(inside.sum()))
            aa.append(area)
        return map(np.asarray, (ds, pf, vv, aa))

    dsw, pw, vw, aw = scan(cw, dew)
    dsc, pc, _, _ = scan(cc, dec)
    rho_p, rho_v, rho_a = pearson(dsw, pw), pearson(dsw, vw), pearson(dsw, aw)
    grow_w = float(dsw[-1] / dsw[0])
    grow_c = float(dsc[-1] / dsc[0])
    c_vol = bool(abs(rho_p) > 0.95 and grow_w > 1.30)
    c_va = bool(abs(rho_v) > abs(rho_a))
    c_comp = bool(grow_c < 1.15)

    xs = np.linspace(-LBOX, LBOX, NBOX)
    h = float(xs[1] - xs[0])
    scale = 2.0 * LBOX / (N - 1)
    rhs = np.zeros((NBOX, NBOX, NBOX))
    sx, sy, sz = SRC
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                px, py, pz = (x - sx) * scale, (y - sy) * scale, (z - sz) * scale
                if max(abs(px), abs(py), abs(pz)) >= LBOX - 0.5 * h:
                    continue
                ix = int(np.argmin(np.abs(xs - px)))
                iy = int(np.argmin(np.abs(xs - py)))
                iz = int(np.argmin(np.abs(xs - pz)))
                if min(ix, iy, iz) <= 0 or max(ix, iy, iz) >= NBOX - 1:
                    continue
                rhs[ix, iy, iz] += 4.0 * np.pi * GCONST * dew[i] / (h**3)
    m = NBOX - 2
    fhat = rhs[1:-1, 1:-1, 1:-1].copy()
    for axv in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=axv)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for axv in (0, 1, 2):
        phat = dst(phat, type=1, axis=axv)
    phi = np.zeros((NBOX, NBOX, NBOX))
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)

    def interp(field, p):
        def one(axg, val):
            k = int(np.searchsorted(axg, val) - 1)
            k = max(0, min(k, len(axg) - 2))
            return k, (val - axg[k]) / (axg[k + 1] - axg[k])

        i, tx = one(xs, p[0])
        jy, ty = one(xs, p[1])
        k, tz = one(xs, p[2])
        acc = 0.0
        for di, wi in ((0, 1 - tx), (1, tx)):
            for dj, wj in ((0, 1 - ty), (1, ty)):
                for dk, wk in ((0, 1 - tz), (1, tz)):
                    acc += wi * wj * wk * field[i + di, jy + dj, k + dk]
        return float(acc)

    accs = [interp(ax, (r, 0.0, 0.0)) for r in PROBES]
    lr = np.log(np.array(PROBES, float))
    la = np.log(np.abs(np.array(accs, float)))
    slope = float(np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)[0][0])
    c_lin = bool(abs(slope - 1.0) < 0.40 and abs(slope - 1.0) < abs(slope + 2.0) and all(a < 0 for a in accs))
    payload = {
        "task": "m9.41_audit_uniform",
        "rho_P": rho_p,
        "rho_V": rho_v,
        "rho_A": rho_a,
        "grow_wide": grow_w,
        "grow_comp": grow_c,
        "a_r": accs,
        "slope": slope,
        "C_fl_grow": c_vol,
        "C_VA_diagnostic": c_va,
        "C_compact": c_comp,
        "C_lin": c_lin,
        "verdicts": {
            "C_fl_grow": "CONFIRMED" if c_vol else "REFUTED",
            "C_lin": "CONFIRMED" if c_lin else "REFUTED",
            "C_compact": "CONFIRMED" if c_comp else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_41_audit_uniform.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c_vol else 1


if __name__ == "__main__":
    raise SystemExit(main())
